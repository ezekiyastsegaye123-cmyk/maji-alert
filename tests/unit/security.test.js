/**
 * Security & Red-Team Command Injection Test Suite
 * =================================================
 * Verifies that malicious inputs, shell metacharacters, and command chaining
 * payloads are neutralized by strict Zod schema validation and safe non-shell spawn().
 */

const fs = require('fs');
const path = require('path');
const { safeValidatePredictionInput } = require('../../src/validation/predictionInput');
const { executePrediction } = require('../../src/services/mlService');

describe('Security Red-Team: Command Injection Neutralization', () => {
  const canaryFile = path.join(__dirname, 'canary_pwned.txt');

  afterEach(() => {
    if (fs.existsSync(canaryFile)) {
      fs.unlinkSync(canaryFile);
    }
  });

  const injectionPayloads = [
    '; rm -rf /',
    '; touch ' + canaryFile,
    '$(touch ' + canaryFile + ')',
    '`touch ' + canaryFile + '`',
    '| cat /etc/passwd',
    '& whoami &',
    '&& echo pwned',
    '4.88; id',
    '4.88\nreboot',
    '\' OR 1=1 --',
    '{"$gt": ""}',
    '../../../../etc/shadow',
    '${IFS}reboot',
  ];

  describe('Boundary Layer 1: Zod Schema Rejection', () => {
    injectionPayloads.forEach((payload) => {
      it(`strictly rejects malicious latitude injection: "${payload}"`, () => {
        const result = safeValidatePredictionInput({
          latitude: payload,
          longitude: 38.08,
          year: 2026,
        });
        expect(result.success).toBe(false);
        // Ensure no file was created by shell execution
        expect(fs.existsSync(canaryFile)).toBe(false);
      });

      it(`strictly rejects malicious longitude injection: "${payload}"`, () => {
        const result = safeValidatePredictionInput({
          latitude: 4.88,
          longitude: payload,
          year: 2026,
        });
        expect(result.success).toBe(false);
        expect(fs.existsSync(canaryFile)).toBe(false);
      });

      it(`strictly rejects malicious year injection: "${payload}"`, () => {
        const result = safeValidatePredictionInput({
          latitude: 4.88,
          longitude: 38.08,
          year: payload,
        });
        expect(result.success).toBe(false);
        expect(fs.existsSync(canaryFile)).toBe(false);
      });
    });
  });

  describe('Boundary Layer 2: Non-Shell Process Invocation (spawn with shell:false)', () => {
    it(
      'does not evaluate shell metacharacters even if passed directly to process wrapper',
      async () => {
        // Even if attacker somehow passes a string with shell operators directly to executePrediction:
        try {
          await executePrediction({
            latitude: `4.88; touch ${canaryFile}`,
            longitude: 38.08,
            year: 2026,
          });
        } catch (err) {
          // Expected to fail in python CLI argument parsing (ValueError or InvalidInputError)
        }

        // CRITICAL ASSERTION: The canary file must NOT have been touched by a shell!
        expect(fs.existsSync(canaryFile)).toBe(false);
      },
      20000
    );
  });

  describe('Boundary Layer 3: NoSQL / Type Confusion Injections', () => {
    it('rejects MongoDB operator injection objects in latitude/longitude', () => {
      const result = safeValidatePredictionInput({
        latitude: { $gt: 0 },
        longitude: 38.08,
        year: 2026,
      });
      expect(result.success).toBe(false);
    });

    it('rejects array-based parameters', () => {
      const result = safeValidatePredictionInput({
        latitude: [4.88],
        longitude: [38.08],
        year: 2026,
      });
      expect(result.success).toBe(false);
    });
  });
});
