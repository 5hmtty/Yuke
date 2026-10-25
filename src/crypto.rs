use anyhow::Context;

/// File-header magic (6 bytes) -- the common prefix of every encrypted format, detected automatically on the reading side
pub const MAGIC: &[u8; 6] = b"VIQENC";

const MODE_EMBEDDED: u8 = 4; // v2: the embedded master key derives a per-file key (all new writes use this mode)


// Detection

/// Whether these bytes are in an encrypted format (magic detection -- file names never change: a wav stays .wav, a json stays .json)
pub fn is_encrypted(blob: &[u8]) -> bool {
    blob.len() >= 7 && &blob[..6] == MAGIC
}

// Random source (BCryptGenRandom, the system-preferred RNG)

fn fill_random(buf: &mut [u8]) -> anyhow::Result<()> {
    use windows::Win32::Security::Cryptography::{
        BCRYPT_USE_SYSTEM_PREFERRED_RNG, BCryptGenRandom,
    };
    let status = unsafe { BCryptGenRandom(None, buf, BCRYPT_USE_SYSTEM_PREFERRED_RNG) };
    anyhow::ensure!(status.is_ok(), "BCryptGenRandom failed to produce randomness: {status:?}");
    Ok(())
}

// Master key (obfuscated constant, distributed inside the binary)

const MASTER_OBF: [u8; 32] = [
    47, 149, 14, 207, 113, 244, 133, 16, 102, 197, 69, 129, 241, 69, 192, 140, 220, 250, 250, 63,
    215, 248, 25, 31, 209, 7, 7, 145, 66, 113, 152, 122,
];
const MASTER_MASK: [u8; 32] = [
    17, 64, 111, 158, 205, 252, 43, 90, 137, 184, 231, 22, 69, 116, 163, 210, 1, 48, 95, 142, 189,
    236, 27, 74, 121, 168, 215, 6, 53, 100, 147, 194,
];

fn master_key() -> [u8; 32] {
    let mut k = MASTER_OBF;
    for i in 0..32 {
        k[i] ^= MASTER_MASK[i];
    }
    k
}

/// v2 per-file key derivation: AES-256(master_key, file_nonce) -- CTR the master key over the nonce
/// (counter blocks 0/1); the resulting 32-byte keystream is that file's AES key. The nonce is stored in the header
/// in plaintext (its role equals a public key input; the only secret is the master key itself).
fn derive_file_key(master: &[u8; 32], nonce: &[u8; 16]) -> [u8; 32] {
    aes256_ctr(master, nonce, &[0u8; 32])
        .try_into()
        .expect("32-byte keystream")
}

// Public encode / decode

pub fn protect(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut nonce = [0u8; 16];
    fill_random(&mut nonce)?;
    let fk = derive_file_key(&master_key(), &nonce);
    let ct = aes256_ctr(&fk, &nonce, data);
    let mut out = Vec::with_capacity(6 + 1 + 16 + ct.len());
    out.extend_from_slice(MAGIC);
    out.push(MODE_EMBEDDED);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ct);
    Ok(out)
}

/// Decrypt (embedded-key tier; plaintext input is returned as-is)
pub fn unprotect(blob: &[u8]) -> anyhow::Result<Vec<u8>> {
    if !is_encrypted(blob) {
        return Ok(blob.to_vec());
    }
    anyhow::ensure!(blob.len() >= 7 + 16, "ciphertext length out of range (file truncated or corrupt)");
    let mode = blob[6];
    anyhow::ensure!(mode == MODE_EMBEDDED, "unknown encryption mode byte {mode} (corrupt or incompatible version)");
    let nonce: [u8; 16] = blob[7..23].try_into().unwrap();
    let fk = derive_file_key(&master_key(), &nonce);
    Ok(aes256_ctr(&fk, &nonce, &blob[23..]))
}

/// Read a file and decrypt automatically (magic detection; plaintext files pass through -- the single entry point for dual-compatible reads)
pub fn read_file_auto(path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    let raw = std::fs::read(path).with_context(|| format!("failed to read: {}", path.display()))?;
    if is_encrypted(&raw) {
        unprotect(&raw).with_context(|| format!("failed to decrypt: {}", path.display()))
    } else {
        Ok(raw)
    }
}

/// Read a text file and decrypt automatically (read-side entry such as embed_proto_v2.json)
pub fn read_text_auto(path: &std::path::Path) -> anyhow::Result<String> {
    let bytes = read_file_auto(path)?;
    String::from_utf8(bytes).with_context(|| format!("not valid UTF-8 text: {}", path.display()))
}

// AES-256-CTR (self-implemented, forward cipher plus counter only)

/// AES S-box (FIPS-197 figure 3)
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16,
];

/// AES-256 (Nk=8, Nr=14), forward only. State is a flat 16 bytes: block[4*col + row].
struct Aes256 {
    /// 60 four-byte words (14 rounds + initial = 15 round keys × 4 words)
    w: [[u8; 4]; 60],
}

fn xtime(x: u8) -> u8 {
    (x << 1) ^ (((x >> 7) & 1) * 0x1b)
}

impl Aes256 {
    fn new(key: &[u8; 32]) -> Self {
        let mut w = [[0u8; 4]; 60];
        for i in 0..8 {
            w[i].copy_from_slice(&key[i * 4..i * 4 + 4]);
        }
        // rcon[i/8]; AES-256 needs rcon[1..=7]
        const RCON: [u8; 8] = [0, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40];
        for i in 8..60 {
            let mut t = w[i - 1];
            let r = i % 8;
            if r == 0 {
                t = [
                    SBOX[t[1] as usize] ^ RCON[i / 8],
                    SBOX[t[2] as usize],
                    SBOX[t[3] as usize],
                    SBOX[t[0] as usize],
                ];
            } else if r == 4 {
                t = [
                    SBOX[t[0] as usize],
                    SBOX[t[1] as usize],
                    SBOX[t[2] as usize],
                    SBOX[t[3] as usize],
                ];
            }
            for j in 0..4 {
                w[i][j] = w[i - 8][j] ^ t[j];
            }
        }
        Aes256 { w }
    }

    fn round_key(&self, round: usize) -> impl Fn(usize, usize) -> u8 + '_ {
        move |col: usize, row: usize| self.w[4 * round + col][row]
    }

    /// Single-block encryption (FIPS-197 5.1)
    fn encrypt_block(&self, block: &[u8; 16], out: &mut [u8; 16]) {
        let mut s = *block;
        // AddRoundKey(0)
        let rk = self.round_key(0);
        for c in 0..4 {
            for r in 0..4 {
                s[4 * c + r] ^= rk(c, r);
            }
        }
        for round in 1..=14 {
            // SubBytes
            for b in s.iter_mut() {
                *b = SBOX[*b as usize];
            }
            // ShiftRows: row r shifts left by r (block[4*col + row] layout)
            let mut t = [0u8; 16];
            for r in 0..4 {
                for c in 0..4 {
                    t[4 * c + r] = s[4 * ((c + r) % 4) + r];
                }
            }
            s = t;
            // MixColumns (skipped in the final round)
            if round < 14 {
                let mut m = [0u8; 16];
                for c in 0..4 {
                    let a0 = s[4 * c];
                    let a1 = s[4 * c + 1];
                    let a2 = s[4 * c + 2];
                    let a3 = s[4 * c + 3];
                    m[4 * c] = xtime(a0) ^ (xtime(a1) ^ a1) ^ a2 ^ a3;
                    m[4 * c + 1] = a0 ^ xtime(a1) ^ (xtime(a2) ^ a2) ^ a3;
                    m[4 * c + 2] = a0 ^ a1 ^ xtime(a2) ^ (xtime(a3) ^ a3);
                    m[4 * c + 3] = (xtime(a0) ^ a0) ^ a1 ^ a2 ^ xtime(a3);
                }
                s = m;
            }
            // AddRoundKey(round)
            let rk = self.round_key(round);
            for c in 0..4 {
                for r in 0..4 {
                    s[4 * c + r] ^= rk(c, r);
                }
            }
        }
        *out = s;
    }
}

/// AES-256-CTR: the nonce is the initial counter (128-bit, big-endian increment). Encryption equals decryption.
fn aes256_ctr(key: &[u8; 32], nonce: &[u8; 16], data: &[u8]) -> Vec<u8> {
    let enc = Aes256::new(key);
    let mut ctr = *nonce;
    let mut out = Vec::with_capacity(data.len());
    for chunk in data.chunks(16) {
        let mut ks = [0u8; 16];
        enc.encrypt_block(&ctr, &mut ks);
        for (i, b) in chunk.iter().enumerate() {
            out.push(b ^ ks[i]);
        }
        for i in (0..16).rev() {
            ctr[i] = ctr[i].wrapping_add(1);
            if ctr[i] != 0 {
                break;
            }
        }
    }
    out
}

// Unit tests (NIST vectors + round-trip + error tolerance)

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// FIPS-197 appendix C.3 AES-256 single-block vector
    #[test]
    fn aes256_fips197_vector() {
        let key: [u8; 32] = hex("000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f")
            .try_into()
            .unwrap();
        let pt: [u8; 16] = hex("00112233445566778899aabbccddeeff").try_into().unwrap();
        let expect = hex("8ea2b7ca516745bfeafc49904b496089");
        let enc = Aes256::new(&key);
        let mut out = [0u8; 16];
        enc.encrypt_block(&pt, &mut out);
        assert_eq!(out.to_vec(), expect);
    }

    /// NIST SP 800-38A F.5.5 AES-256-CTR vectors for all 4 blocks
    #[test]
    fn aes256_ctr_sp800_vector() {
        let key: [u8; 32] = hex("603deb1015ca71be2b73aef0857d77811f352c073b6108d72d9810a30914dff4")
            .try_into()
            .unwrap();
        let nonce: [u8; 16] = hex("f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff").try_into().unwrap();
        let pt = hex("6bc1bee22e409f96e93d7e117393172a\
             ae2d8a571e03ac9c9eb76fac45af8e51\
             30c81c46a35ce411e5fbc1191a0a52ef\
             f69f2445df4f9b17ad2b417be66c3710");
        let expect: Vec<u8> = hex("601ec313775789a5b7a7f504bbf3d228\
             f443e3ca4d62b59aca84e990cacaf5c5\
             2b0930daa23de94ce87017ba2d84988d\
             dfc9c58db67aada613c2dd08457941a6");
        let ct = aes256_ctr(&key, &nonce, &pt);
        assert_eq!(ct, expect);
        // CTR symmetry: encrypting again decrypts
        let back = aes256_ctr(&key, &nonce, &ct);
        assert_eq!(back, pt);
    }

    /// L1 small files: embedded master-key derivation round-trip (not bound to the Windows account)
    #[test]
    fn roundtrip_embedded_small() {
        // Master-key constant guard: OBF^MASK must restore exactly the value pinned at generation time
        let expect = hex("3ed56151bc08ae4aef7da297b431635eddcaa5b16a140255a8afd09777150bb8");
        assert_eq!(master_key().to_vec(), expect, "MASTER_OBF/MASK was modified");
        // Derivation determinism: same nonce gives the same key, different nonces give different keys, and the derived value is not the master key
        let mk = master_key();
        let n1 = [1u8; 16];
        assert_eq!(derive_file_key(&mk, &n1), derive_file_key(&mk, &n1));
        assert_ne!(derive_file_key(&mk, &n1), derive_file_key(&mk, &[2u8; 16]));
        assert_ne!(derive_file_key(&mk, &n1), mk);

        let data: Vec<u8> = (0..100_000u32).map(|i| (i % 251) as u8).collect();
        let enc = protect(&data).unwrap();
        assert!(is_encrypted(&enc));
        assert_eq!(enc[6], MODE_EMBEDDED, "new writes must use the embedded-key mode");
        assert_ne!(enc[..], data[..data.len().min(enc.len())]);
        let dec = unprotect(&enc).unwrap();
        assert_eq!(dec, data);
    }

    /// Magic detection: plaintext wav (RIFF) and json ('{') are not misdetected; encrypted files always match
    #[test]
    fn magic_detection() {
        assert!(!is_encrypted(b"RIFFxxxxWAVEfmt "));
        assert!(!is_encrypted(b"{\n  \"cmd\": []\n}"));
        assert!(!is_encrypted(b""));
        assert!(!is_encrypted(b"VIQEN")); // fewer than 7 bytes
        let enc = protect(b"x").unwrap();
        assert!(is_encrypted(&enc));
    }

    /// Corruption tolerance: truncation or a bad mode byte -> error rather than panic; plaintext passes through
    #[test]
    fn corrupt_inputs_error_not_panic() {
        let data = b"0123456789abcdef".repeat(64);
        let enc = protect(&data).unwrap();
        // Truncated below 7 bytes -> no complete magic -> passes through as plaintext (dual-compatible semantics, no error)
        for cut in [0usize, 6] {
            assert_eq!(unprotect(&enc[..cut]).unwrap(), enc[..cut], "cut={cut}");
        }
        // Magic present but the payload is truncated (v2 header = 6+1+16 bytes) -> must error
        for cut in [7usize, 10, 22] {
            assert!(unprotect(&enc[..cut]).is_err(), "cut={cut}");
        }
        // Bad mode byte
        let mut bad2 = enc.clone();
        bad2[6] = 0xEE;
        assert!(unprotect(&bad2).is_err());
        // Non-encrypted input passes through (no error)
        assert_eq!(unprotect(&data).unwrap(), data);
    }


}
