// tiny-shakespeare-char data utilities. Mirrors nanoGPT's get_batch + char
// tokenizer. Loaded once per page from /m18/tinyshakespeare.txt and split
// 90/10 train/val.

export interface CorpusBundle {
  trainIds: Int32Array;
  valIds: Int32Array;
  vocabSize: number;
  vocab: string;            // sorted unique characters; index = id
}

export async function loadTinyShakespeare(url = '/m18/tinyshakespeare.txt'): Promise<CorpusBundle> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to fetch ${url}: ${res.status}`);
  const text = await res.text();
  // Build vocab — sorted by character code, like nanoGPT's
  // `chars = sorted(list(set(text)))`.
  const set = new Set<string>(); for (const c of text) set.add(c);
  const vocab = Array.from(set).sort().join('');
  if (vocab.length !== 65) {
    throw new Error(`tiny-shakespeare vocab expected 65 chars, got ${vocab.length}`);
  }
  const stoi = new Map<string, number>();
  for (let i = 0; i < vocab.length; i++) stoi.set(vocab[i], i);

  const ids = new Int32Array(text.length);
  for (let i = 0; i < text.length; i++) ids[i] = stoi.get(text[i])!;
  const n = Math.floor(0.9 * ids.length);
  return { trainIds: ids.subarray(0, n), valIds: ids.subarray(n), vocabSize: vocab.length, vocab };
}

// Random-offset batch sampler. `data` is a flat Int32Array of token ids.
// Produces (x, y) where y is x shifted by 1, contiguous of length T per row.
export function getBatch(
  data: Int32Array, B: number, T: number, rng: () => number,
): { x: Int32Array; y: Int32Array } {
  const x = new Int32Array(B * T);
  const y = new Int32Array(B * T);
  const range = data.length - T - 1;
  for (let b = 0; b < B; b++) {
    const ix = Math.floor(rng() * range);
    for (let t = 0; t < T; t++) {
      x[b * T + t] = data[ix + t];
      y[b * T + t] = data[ix + t + 1];
    }
  }
  return { x, y };
}
