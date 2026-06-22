require('dotenv').config({ path: require('path').join(__dirname, '../.env') });
const mongoose = require('mongoose');
const ColonyProfile = require('../models/ColonyProfile');
const { presetFromDetection } = require('../utils/colonyProfilePreset');

const profiles = [
  {
    name: 'Anarobic Count Film',
    description: 'Used for detecting and quantifying anaerobic bacteria that grow in oxygen-free environments.',
    image_path: '../../assets/public/Trays/AC.png',
    icon: 'anaerobic',
    user: null,
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Incubation', value: '37°C' },
      { label: 'Duration',   value: '48 hrs' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 15,
      min_radius: 3,
      max_radius: 50,
      enable_color_grouping: false,
      coarseness: 10,
      neighbours: 10,
    }),
  },
  {
    name: 'Coliform (CC) Film',
    description: 'Designed to detect coliform bacteria, commonly used for testing water and food safety.',
    image_path: '../../assets/public/Trays/CF.png',
    icon: 'coliform',
    user: null,
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Incubation', value: '35°C' },
      { label: 'Duration',   value: '24 hrs' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 15,
      min_radius: 3,
      max_radius: 50,
      enable_color_grouping: false,
      coarseness: 10,
      neighbours: 10,
    }),
  },
  {
    name: 'MacConkey Plates',
    description: 'Selective medium for isolating and differentiating Gram-negative bacteria based on lactose fermentation.',
    image_path: '../../assets/public/Trays/MacConkey.png',
    icon: 'maconkey',
    user: null,
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Medium',    value: 'Agar Plate' },
      { label: 'Indicator', value: 'Neutral Red' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 15,
      min_radius: 3,
      max_radius: 50,
      enable_color_grouping: true,
      coarseness: 10,
      neighbours: 10,
    }),
  },
  {
    name: 'Nutrient Plates',
    description: 'General-purpose medium supporting the growth of a wide range of non-fastidious organisms.',
    image_path: '../../assets/public/Trays/NP.png',
    icon: 'nutrient',
    user: null,
    source: 'lab',
    validated: true,
    parameters: [
      { label: 'Medium',   value: 'Agar Plate' },
      { label: 'Use case', value: 'General growth' },
    ],
    params: presetFromDetection({
      threshold_type: 'regular',
      threshold_value: 5,
      min_radius: 4,
      max_radius: 185,
      enable_color_grouping: false,
      coarseness: 10,
      neighbours: 10,
    }),
  },
];

async function seed() {
  await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/incucount');
  console.log('MongoDB connected');

  let inserted = 0;
  let skipped  = 0;

  for (const data of profiles) {
    const exists = await ColonyProfile.exists({ name: data.name });
    if (exists) {
      console.log(`  SKIP  "${data.name}" (already exists)`);
      skipped++;
    } else {
      await ColonyProfile.create(data);
      console.log(`  INSERT "${data.name}"`);
      inserted++;
    }
  }

  console.log(`\nDone — ${inserted} inserted, ${skipped} skipped.`);
  await mongoose.disconnect();
}

seed().catch(err => {
  console.error(err);
  process.exit(1);
});
