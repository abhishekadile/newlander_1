const { spawn } = require('child_process');
const fs = require('fs-extra');
const path = require('path');
const sharp = require('sharp');

class ColonyDetector {
  constructor() {
    this.coreEnginePath = path.join(process.cwd(), 'core_engine');
    // OpenCFU binary selection:
    // - Windows dev bundles typically ship `core_engine/opencfu.exe`
    // - Linux/Docker typically uses `opencfu` (on PATH) or `core_engine/opencfu`
    // You can override with env: OPENCFU_BIN=/absolute/path/to/opencfu (or just "opencfu")
    this.opencfuPath = this.resolveOpenCFUBinary();
    this.csvOutputPath = path.join(this.coreEnginePath, 'bacterial_colonies.csv');
    this.preprocessScriptPath = path.join(__dirname, 'preprocess.py');
  }

  resolveOpenCFUBinary() {
    const override = (process.env.OPENCFU_BIN || '').trim();
    if (override) return override;

    // Strict OS detection
    if (process.platform === 'win32') {
      const candidate = path.join(this.coreEnginePath, 'opencfu.exe');
      if (fs.existsSync(candidate)) return candidate;
      // Fallback for Windows if exe is missing in core_engine
      return 'opencfu';
    }

    // Linux/Docker/Mac:
    // We expect opencfu to be installed system-wide (e.g. via apt-get in Docker)
    // We DO NOT check core_engine for linux binaries as they are likely not there or incompatible.
    return 'opencfu';
  }

  async detectColonies(imagePath, params = {}) {
    let preprocessResult = null;
    try {
      // 1. Validate
      if (!fs.existsSync(imagePath)) {
        return { success: false, error: 'Image file does not exist' };
      }

      // 2. Get image dimensions (for frontend info only)
      let imageWidth = 0, imageHeight = 0;
      try {
        const meta = await sharp(imagePath).metadata();
        imageWidth = meta.width;
        imageHeight = meta.height;
      } catch (e) {
        console.warn('Could not read image metadata:', e.message);
      }
      console.log('imageWidth', imageWidth);
      // 3. Preprocess (IncuCount-compatible ROI + scaling)
      preprocessResult = await this.preprocessImage(imagePath);

      const processedImagePath = preprocessResult?.processedImagePath || imagePath;
      console.log('processedImagePath', processedImagePath);
      // 4. Run OpenCFU on processed image
      const detectionResult = await this.runOpenCFU(processedImagePath, params);

      if (!detectionResult.success) {
        return detectionResult;
      }

      // 5. Parse Results (OpenCFU outputs in processed/cropped space)
      const coloniesProcessed = await this.getColoniesFromDetectionResult(detectionResult);
      let colonies = this.transformColoniesToOriginalSpace(
        coloniesProcessed,
        preprocessResult?.meta
      );

      // Final safety: enforce detected dish circle in original coordinates.
      // (In IncuCount/OpenCFU this is normally handled by the ROI mask filter step.)
      colonies = this.filterColoniesToDetectedCircle(colonies, preprocessResult?.meta);

      const detectedCircle = preprocessResult?.meta?.detected_circle || { present: false };
      const detectedCircleProcessed = preprocessResult?.meta?.detected_circle_processed || { present: false };
      console.log('detectedCircle', detectedCircle);
      console.log('detectedCircleProcessed', detectedCircleProcessed);

      return {
        success: true,
        colonies: colonies,
        colonyCount: colonies.length,
        imageDimensions: { width: imageWidth, height: imageHeight },
        processedImagePath,
        originalImagePath: imagePath,
        meta: preprocessResult?.meta || null,
        resultSource: detectionResult.resultSource || null,
        detectedCircle,
        detectedCircleProcessed
      };

    } catch (error) {
      console.error('Error detecting colonies:', error);
      return { success: false, error: error.message };
    }
  }

  async preprocessImage(imagePath) {
    // Default: no-op preprocessing
    const fallback = {
      processedImagePath: imagePath,
      cleanupPaths: [],
      meta: {
        cropped: false,
        scale_factor: 1,
        crop_offset: { x: 0, y: 0 },
        crop_offset_processed: { x: 0, y: 0 },
        detected_circle: { present: false },
        detected_circle_processed: { present: false }
      }
    };

    try {
      if (!(await fs.pathExists(this.preprocessScriptPath))) {
        console.warn('preprocess.py not found, skipping preprocessing:', this.preprocessScriptPath);
        return fallback;
      }

      const result = await this.runPythonPreprocess(imagePath);
      if (!result?.processedImagePath) return fallback;
      return result;
    } catch (e) {
      console.warn('Preprocessing failed, using original image:', e?.message || String(e));
      return fallback;
    }
  }

  runPythonPreprocess(imagePath) {
    return new Promise((resolve, reject) => {
      // Use python3 on non-windows systems (e.g. Docker)
      const defaultPython = process.platform === 'win32' ? 'python' : 'python3';
      const pythonBin = process.env.PYTHON_BIN || defaultPython;

      const args = [
        this.preprocessScriptPath,
        '--image',
        path.resolve(imagePath)
      ];

      const proc = spawn(pythonBin, args, {
        cwd: __dirname,
        stdio: ['ignore', 'pipe', 'pipe']
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (d) => (stdout += d.toString()));
      proc.stderr.on('data', (d) => (stderr += d.toString()));

      proc.on('error', (err) => reject(err));

      proc.on('close', async (code) => {
        if (code !== 0) {
          reject(new Error(`preprocess.py failed (code ${code}): ${stderr || stdout}`));
          return;
        }
        let parsed;
        try {
          parsed = JSON.parse(stdout);
        } catch (e) {
          reject(new Error(`Failed to parse preprocess.py JSON: ${e.message}. stdout=${stdout} stderr=${stderr}`));
          return;
        }

        const processedImagePath = parsed?.processed_image_path;
        if (!processedImagePath) {
          reject(new Error(`preprocess.py returned no processed_image_path. stdout=${stdout}`));
          return;
        }

        // Ensure file exists before returning
        const exists = await fs.pathExists(processedImagePath);
        if (!exists) {
          reject(new Error(`Processed image not found: ${processedImagePath}`));
          return;
        }

        const meta = {
          cropped: !!parsed.cropped,
          scale_factor: typeof parsed.scale_factor === 'number' ? parsed.scale_factor : 1,
          crop_offset: parsed.crop_offset || { x: 0, y: 0 },
          crop_offset_processed: parsed.crop_offset_processed || { x: 0, y: 0 },
          detected_circle: parsed.detected_circle || { present: false },
          detected_circle_processed: parsed.detected_circle_processed || { present: false },
          original_size: parsed.original_size || null,
          processed_size: parsed.processed_size || null,
          upscaled_size: parsed.upscaled_size || null
        };

        resolve({
          processedImagePath,
          cleanupPaths: [processedImagePath],
          meta
        });
      });
    });
  }

  transformColoniesToOriginalSpace(colonies, meta) {
    if (!Array.isArray(colonies) || colonies.length === 0) return colonies || [];

    const scale = meta && Number(meta.scale_factor) > 0 ? Number(meta.scale_factor) : 1;
    const offP = meta && meta.crop_offset_processed ? meta.crop_offset_processed : { x: 0, y: 0 };
    const offX = Number(offP.x) || 0;
    const offY = Number(offP.y) || 0;

    return colonies.map((c) => {
      const out = { ...c };
      const x = Number(out.x) || 0;
      const y = Number(out.y) || 0;
      const r = Number(out.radius) || 0;

      // OpenCFU returns coordinates in processed/cropped space.
      // To map to original:
      // 1) add crop offset in processed coordinates
      // 2) divide by scale_factor (upscale-to-6000 behavior)
      out.x = (x + offX) / scale;
      out.y = (y + offY) / scale;
      out.radius = r / scale;

      if (typeof out.area === 'number') {
        out.area = out.area / (scale * scale);
      }

      return out;
    });
  }

  filterColoniesToDetectedCircle(colonies, meta) {
    if (!Array.isArray(colonies) || colonies.length === 0) return colonies || [];
    const circle = meta && meta.detected_circle ? meta.detected_circle : null;
    if (!circle || !circle.present || !circle.center || !Number.isFinite(circle.radius)) return colonies;

    const cx = Number(circle.center.x);
    const cy = Number(circle.center.y);
    const r = Number(circle.radius);
    if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(r) || r <= 0) return colonies;

    // Small tolerance to avoid removing edge-touching colonies due to rounding.
    const tol = 1.02;
    const r2 = (r * tol) * (r * tol);

    return colonies.filter((colony) => {
      const x = Number(colony.x);
      const y = Number(colony.y);
      if (!Number.isFinite(x) || !Number.isFinite(y)) return false;
      const dx = x - cx;
      const dy = y - cy;
      return (dx * dx + dy * dy) <= r2;
    });
  }

  async runOpenCFU(imagePath, params) {
    return new Promise((resolve) => {
      (async () => {
        try {
        // If opencfuPath is a real filesystem path, verify it exists.
        // If it's a PATH command (e.g., "opencfu"), let spawn() resolve it.
        const looksLikePath =
          path.isAbsolute(this.opencfuPath) || this.opencfuPath.includes(path.sep);
        if (looksLikePath && !fs.existsSync(this.opencfuPath)) {
          resolve({ success: false, error: `OpenCFU executable not found at: ${this.opencfuPath}` });
          return;
        }

        // Prevent stale results: OpenCFU writes to a fixed filename (bacterial_colonies.csv).
        // This repo also ships a pre-existing CSV, so we must delete it before each run.
        // We also track the previous stat so we can detect "unchanged" output.
        let prevStat = null;
        try {
          if (await fs.pathExists(this.csvOutputPath)) {
            prevStat = await fs.stat(this.csvOutputPath);
            await fs.remove(this.csvOutputPath);
          }
        } catch (_) { }

        const args = [
          '-i', path.resolve(imagePath),
          '-d', params.threshold_type === 'inverted' ? 'inv' : 'reg',
          '-t', String(params.threshold_value || 15),
          '-r', String(params.min_radius || 3),
          '-R', String(params.max_radius || 50)
        ];

        if (params.enable_color_grouping) {
          args.push(`-D${params.coarseness || 10.0}`);
          args.push(`-N${params.neighbours || 10}`);
        }

        const opencfu = spawn(this.opencfuPath, args, {
          cwd: this.coreEnginePath,
          stdio: ['ignore', 'pipe', 'pipe']
        });

        let stdout = '';
        let stderr = '';

        opencfu.stdout.on('data', (d) => { stdout += d.toString(); });
        opencfu.stderr.on('data', (d) => stderr += d.toString());

        opencfu.on('close', async (code) => {
          if (code === 0) {
            const csvExists = await fs.pathExists(this.csvOutputPath);
            if (csvExists) {
              let csvFresh = true;
              try {
                const st = await fs.stat(this.csvOutputPath);
                if (prevStat && st.mtimeMs <= prevStat.mtimeMs && st.size === prevStat.size) {
                  csvFresh = false;
                }
              } catch (_) { }

              resolve({
                success: true,
                csvPath: this.csvOutputPath,
                stdout,
                stderr,
                resultSource: csvFresh ? 'csv' : 'stdout'
              });
              return;
            }

            // Fallback: OpenCFU prints the header + rows to stdout.
            // If for any reason the CSV isn't created in our expected working directory,
            // still allow the request to succeed based on stdout parsing.
            resolve({ success: true, csvPath: null, stdout, stderr, resultSource: 'stdout' });
            return;
          }

          resolve({ success: false, error: `OpenCFU failed (code ${code}): ${stderr}` });
        });

        opencfu.on('error', (err) => resolve({ success: false, error: err.message }));

        } catch (error) {
          resolve({ success: false, error: error.message });
        }
      })();
    });
  }

  async getColoniesFromDetectionResult(detectionResult) {
    // Prefer CSV when present; fall back to stdout parsing.
    if (detectionResult?.csvPath && (await fs.pathExists(detectionResult.csvPath))) {
      const colonies = await this.parseDetectionResults(detectionResult.csvPath);
      if (Array.isArray(colonies) && colonies.length > 0) return colonies;
      // If CSV exists but is empty/unparseable, fall back to stdout.
    }

    const stdout = String(detectionResult?.stdout || '');
    const coloniesFromStdout = this.parseDetectionOutputText(stdout);
    return coloniesFromStdout;
  }

  parseDetectionOutputText(text) {
    try {
      const rawLines = String(text || '').split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
      if (rawLines.length === 0) return [];

      // Find the header line OpenCFU prints. It should start with "IsValid,".
      const headerIdx = rawLines.findIndex((l) => /^isvalid\s*,/i.test(l));
      if (headerIdx === -1) return [];

      const lines = rawLines.slice(headerIdx);
      if (lines.length <= 1) return [];

      const headers = lines[0].split(',').map((h) => h.trim().toLowerCase());
      const colonies = [];

      for (let i = 1; i < lines.length; i++) {
        const line = lines[i];
        // Skip non-row lines that can appear in stdout (e.g., "Results saved to ...")
        if (!line.includes(',')) continue;

        const values = line.split(',').map((v) => v.trim());
        if (values.length < headers.length) continue;

        const colony = {};
        headers.forEach((header, idx) => {
          const val = values[idx];
          if (['x', 'y', 'radius', 'area', 'circularity'].includes(header)) {
            colony[header] = parseFloat(val) || 0;
          } else {
            colony[header] = val;
          }
        });

        colonies.push(colony);
      }

      return colonies;
    } catch (e) {
      console.error('Stdout parse error:', e);
      return [];
    }
  }

  async parseDetectionResults(csvPath) {
    try {
      if (!fs.existsSync(csvPath)) return [];
      const content = await fs.readFile(csvPath, 'utf8');
      const lines = content.trim().split('\n');
      if (lines.length <= 1) return [];

      const colonies = [];
      const headers = lines[0].split(',').map(h => h.trim().toLowerCase());

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const colony = {};

        headers.forEach((header, idx) => {
          const val = values[idx];
          if (['x', 'y', 'radius', 'area', 'circularity'].includes(header)) {
            colony[header] = parseFloat(val) || 0;
          } else {
            colony[header] = val;
          }
        });
        colonies.push(colony);
      }
      return colonies;
    } catch (e) {
      console.error('Parse error:', e);
      return [];
    }
  }
}

module.exports = ColonyDetector;
