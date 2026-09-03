const { io } = require('socket.io-client');

async function runSocketE2ETest() {
  console.log('--- Starting Socket.io End-to-End Integration Test ---');
  const socket = io('http://127.0.0.1:3000', {
    transports: ['websocket', 'polling'],
    timeout: 5000,
  });

  await new Promise((resolve, reject) => {
    socket.on('connect', () => {
      console.log('Socket connected successfully with ID:', socket.id);
      resolve();
    });
    socket.on('connect_error', (err) => reject(new Error('Connection error: ' + err.message)));
  });

  // Test 1: Valid prediction
  console.log('Test 1: Emitting drought:predict for Borana Zone (4.88, 38.08, 2026)...');
  let statusReceived = false;
  socket.on('drought:status', (msg) => {
    console.log('Received drought:status:', msg);
    statusReceived = true;
  });

  const resultPromise = new Promise((resolve, reject) => {
    socket.once('drought:prediction_result', (data) => resolve(data));
    socket.once('drought:prediction_error', (err) => reject(err));
    setTimeout(() => reject(new Error('Timed out waiting for socket prediction result')), 10000);
  });

  socket.emit('drought:predict', { latitude: 4.88, longitude: 38.08, year: 2026 });
  const result = await resultPromise;
  console.log('Received drought:prediction_result:', JSON.stringify(result, null, 2));

  if (!statusReceived) throw new Error('Did not receive drought:status event');
  if (![0, 1, 2].includes(result.predicted_drought_class)) throw new Error('Invalid predicted class');
  if (typeof result.model_confidence !== 'number') throw new Error('Missing or invalid model_confidence');
  if (!result.confidence_probabilities) throw new Error('Missing confidence_probabilities');
  console.log('Test 1 PASSED: Valid socket prediction succeeded with confidence:', result.model_confidence);

  // Test 2: Rate limit verification
  console.log('Test 2: Verifying rate limit / rapid duplicate prevention...');
  const rateLimitPromise = new Promise((resolve) => {
    socket.once('drought:prediction_error', (err) => resolve(err));
  });
  socket.emit('drought:predict', { latitude: 4.88, longitude: 38.08, year: 2026 });
  const rateLimitErr = await rateLimitPromise;
  console.log('Received expected rate limit error:', rateLimitErr);
  if (rateLimitErr.code !== 'RATE_LIMITED') throw new Error('Expected RATE_LIMITED code');
  console.log('Test 2 PASSED: Rate limit properly enforced.');

  // Test 3: Invalid parameters validation after cooldown (MIN_COOLDOWN_MS = 2000)
  console.log('Waiting 2200ms for rate-limit cooldown...');
  await new Promise((r) => setTimeout(r, 2200));

  console.log('Test 3: Testing validation error for invalid latitude (95.0)...');
  const errPromise = new Promise((resolve) => {
    socket.once('drought:prediction_error', (err) => resolve(err));
  });
  socket.emit('drought:predict', { latitude: 95.0, longitude: 38.08, year: 2026 });
  const errData = await errPromise;
  console.log('Received validation error as expected:', errData);
  if (errData.code !== 'VALIDATION_ERROR') throw new Error('Expected VALIDATION_ERROR code');
  console.log('Test 3 PASSED: Input validation properly enforced over Socket.io.');

  socket.disconnect();
  console.log('--- All Socket.io E2E Tests PASSED ---');
}

runSocketE2ETest()
  .then(() => process.exit(0))
  .catch((err) => {
    console.error('Socket E2E Test FAILED:', err);
    process.exit(1);
  });
