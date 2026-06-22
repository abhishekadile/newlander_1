# 06 — Overall Findings & Recommendations

## Executive Summary

**IncuCountAPI** is a well-structured, production-oriented backend that successfully wraps OpenCFU into a cloud-ready REST API. The core detection pipeline is scientifically sound and based on published algorithms. However, several **operational and security issues** require attention before scaling to multi-user production use.

---

## Findings

### 🔴 Critical Issues

#### 1. Race Condition on `bacterial_colonies.csv`
- **Location:** `colonyDetector.js` → `runOpenCFU()`
- **Issue:** All OpenCFU processes write to a **single fixed filename** (`core_engine/bacterial_colonies.csv`). Under concurrent requests, request B may read request A's CSV if both finish within a small time window.
- **Impact:** Wrong colony counts returned to users; silent data corruption.
- **Fix:** Use a unique output path per request (e.g., `result_<uuid>.csv`). Requires patching the OpenCFU argument parser or using a per-request temp working directory.

#### 2. Uploaded Files Never Deleted
- **Location:** `server.js`, `POST /detect`
- **Issue:** Multer stores uploaded images in `/uploads/`. The code comments acknowledge this but no cleanup occurs.
- **Impact:** Disk exhaustion in production over time, potentially containing user data (privacy concern).
- **Fix:** Delete uploaded files after detection completes (add `await fs.remove(imagePath)` after `detectColonies()`).

#### 3. Default JWT Secret Exposed
- **Location:** `authController.js` line 5, `authMiddleware.js` line 3
- **Issue:** `const JWT_SECRET = process.env.JWT_SECRET || 'change-me-in-production';`
- **Impact:** If `JWT_SECRET` is not set in the deployment environment, all JWTs are signed with the known string `'change-me-in-production'`, making them trivially forgeable.
- **Fix:** Throw a startup error if `JWT_SECRET` is not set; never fall back to a known default.

---

### 🟡 Medium Issues

#### 4. No Rate Limiting on `/detect`
- **Location:** `server.js`
- **Issue:** The colony detection endpoint accepts unlimited concurrent requests. Each spawns two CPU-intensive child processes (Python + OpenCFU).
- **Impact:** A single client can easily DoS the server by sending multiple rapid requests.
- **Fix:** Add `express-rate-limit` (already in patterns used for `/api/logs`) to `/detect`.

#### 5. Python Process Cold Start (200–500ms per request)
- **Location:** `colonyDetector.js` → `runPythonPreprocess()`
- **Issue:** Python + OpenCV are imported fresh on every request.
- **Impact:** Adds 200–500ms overhead per request; significant at scale.
- **Fix:** Use a persistent Python microservice (FastAPI/Flask) with keep-alive.

#### 6. Preprocessing Temp Files Not Cleaned Up
- **Location:** `preprocess.py` → saves to `temp_preprocessing/preprocessed_<uuid>.png`
- **Issue:** `colonyDetector.js` receives `cleanupPaths` in the preprocessing result but never actually deletes them.
- **Impact:** `temp_preprocessing/` grows indefinitely.
- **Fix:** After detection completes, iterate `preprocessResult.cleanupPaths` and delete.

#### 7. No Authentication on `/detect`
- **Location:** `server.js`
- **Issue:** The primary detection endpoint requires no authentication. Anyone with network access can use the colony detection service.
- **Impact:** Unauthenticated API use; potential for abuse.
- **Fix:** Optionally add `requireAuth` middleware or at minimum a license check before detection.

#### 8. Colony Profile Authorization is Weak
- **Location:** `colonyProfileController.js` → `update()`, `remove()`
- **Issue:** Ownership check falls back to `userId` from the request body if JWT `req.user.sub` is not available:
  ```js
  const isOwner = profile.user && profile.user.toString() === (userId || req.user?.sub);
  ```
  A request with a spoofed `userId` body field and no JWT can potentially satisfy this check.
- **Fix:** Always require a valid JWT for profile mutation; never trust `userId` from the request body for ownership verification.

#### 9. Duplicate Key in `IMAGE_ID_TO_FILENAME`
- **Location:** `server.js` lines 71–72
- **Issue:** `'WIN_20250905_11_44_26_Pro'` key is defined **twice** in the object literal. JavaScript silently uses the last value; the comment even notes this as a "filename mismatch" that doesn't actually exist.
- **Fix:** Remove the duplicate entry.

---

### 🟢 Minor Issues / Observations

#### 10. No Test Suite
- **Issue:** `package.json` has `"test": "echo \"Error: no test specified\""`.
- **Recommendation:** Add integration tests for the `/detect` endpoint and unit tests for coordinate transformation math.

#### 11. CORS Wildcard
- **Location:** `server.js` line 20 — `app.use(cors())`
- **Issue:** CORS is enabled for all origins (`*`) with no restrictions.
- **Recommendation:** Restrict CORS to known frontend origins in production.

#### 12. `test_api.js` Left in Root
- **Issue:** A test script is committed at the root of the repository.
- **Recommendation:** Move to a `tests/` directory or `.gitignore`.

#### 13. Admin Secret in Query String
- **Location:** Multiple admin routes use `?adminSecret=...`
- **Issue:** Query string parameters appear in server access logs, browser history, and HTTP referrer headers.
- **Recommendation:** Move `adminSecret` to request body or `Authorization` header.

#### 14. `assert(!fs.isOpened())` Logic Error in C++
- **Location:** `core_engine/src/processor/src/Processor.cpp` lines 57–59
- **Issue:** The code asserts `!fs.isOpened()` (which means it will crash if the file IS open, but the intent is to crash if it is NOT). This appears to be inverted assertion logic — it should be `assert(fs.isOpened())`.
- **Recommendation:** Fix the assertion direction to correctly detect missing classifier files.

#### 15. Classifier XML Files Not Validated at Startup
- **Issue:** If `trainnedClassifier.xml` is missing or corrupt, OpenCFU silently crashes or produces wrong results. The Node.js wrapper only checks for the binary, not the classifier.
- **Recommendation:** Add a health check that verifies OpenCFU can load its classifiers on startup.

---

## Recommendations Summary

| Priority | Action | Effort |
|----------|--------|--------|
| 🔴 1 | Fix race condition: per-request CSV output path | Medium |
| 🔴 2 | Delete uploaded files after detection | Low |
| 🔴 3 | Require `JWT_SECRET` env var at startup | Low |
| 🟡 4 | Rate limit `/detect` endpoint | Low |
| 🟡 5 | Replace Python spawn with persistent process | High |
| 🟡 6 | Clean up preprocessing temp files | Low |
| 🟡 7 | Add auth to `/detect` (license check) | Medium |
| 🟡 8 | Fix profile ownership check (remove body `userId` trust) | Low |
| 🟡 9 | Remove duplicate image ID key | Low |
| 🟢 10 | Add test suite | High |
| 🟢 11 | Restrict CORS origins | Low |
| 🟢 12 | Move admin secret to header/body | Low |
| 🟢 13 | Fix inverted assert in Processor.cpp | Low |
| 🟢 14 | Validate classifier files on startup | Medium |

---

## Architecture Assessment

### Strengths

| Strength | Detail |
|----------|--------|
| Clean MVC structure | Routes → Controllers → Models well-separated |
| Versioned profile schema | `schema_version` field allows future migrations |
| Backward compatibility | Legacy flat params handled gracefully in `parseProfilePreset()` |
| Docker-ready | Production deployment path via Dockerfile is clean |
| Graceful preprocessing fallback | 3-tier Hough fallback prevents hard failures |
| Coordinate transformation | Correct inverse mapping from processed to original space |

### Technical Debt

| Debt | Detail |
|------|--------|
| No async job model | Synchronous blocking detection per request |
| Single-process bottleneck | Node.js + Python + OpenCFU all blocking per request |
| No metrics/observability | No request timing, no structured logging for detection |
| Magic numbers in preprocessing | `6000`, `196`, `1200`, `0.5` should be named constants |
| Hardcoded demo circles | Should be externalised to a config file |

---

## Technology Stack Summary

| Layer | Technology | Version |
|-------|-----------|---------|
| API runtime | Node.js | 20 LTS |
| Web framework | Express | 4.x |
| Database | MongoDB | Any (via Mongoose 9) |
| File upload | Multer | 1.4.5-lts |
| Image metadata | Sharp | 0.33 |
| Auth | JWT (jsonwebtoken) | 9.x |
| Password hash | bcrypt | 6.x |
| Detection engine | OpenCFU (C++) | ~2.x (fork) |
| CV library (C++) | OpenCV | 4.12 |
| Preprocessing | Python 3 + OpenCV headless | Latest |
| Containerisation | Docker | Debian Bullseye |
| External HTTP | Axios | 1.17 |
| Installer delivery | Google Drive (gdriveInstaller.js) | Custom |
