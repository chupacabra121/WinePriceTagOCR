export const meta = {
  name: 'match-brands',
  description: 'Match each distinct wine name to a Group/Winery/Label in the standardized brand sheet',
  phases: [{ title: 'Match', detail: 'one agent per batch of 40 names, choosing from a shortlist' }],
}

const dir = (args && args.dir) || '/Users/tandrei/WinePriceTagOCR/out/brandmatch'
const batches = (args && args.batches) || []
if (!batches.length) { log('no batches supplied'); return { done: 0 } }

log(`matching ${batches.length} batch(es) of wine names to the brand sheet`)
phase('Match')

const REPORT = {
  type: 'object', additionalProperties: false,
  required: ['written', 'matched', 'unmatched', 'notes'],
  properties: {
    written: { type: 'string', description: 'Absolute path of the answer file written.' },
    matched: { type: 'integer' },
    unmatched: { type: 'integer' },
    notes: { type: 'string', description: 'Anything a human should know. Empty string if nothing.' },
  },
}

const results = await pipeline(batches, (b) => agent(
  `Match wine names to a standardized brand sheet.

Read: ${dir}/batches/${b}.json
Write: ${dir}/answers/${b}.json

The input is a JSON array. Each entry has:
  key         - identifier, echo it back unchanged
  name        - the product name as read off a shop shelf tag, often abbreviated or clipped
  tag         - the raw tag transcription, for extra context
  seen        - how many shelf rows share this name
  candidates  - shortlist of "Group | Winery | Label" strings from the brand sheet,
                retrieved by shared rare words. May be empty, and the right answer
                is NOT guaranteed to be in it.

For each entry decide which candidate is the same brand, and write an array of objects:
  {"key": <echoed>, "choice": <1-based index into candidates, or 0 for none>,
   "confidence": "high"|"medium"|"low", "why": "<short reason>"}

How to judge:
  - Match on BRAND identity - the winery and the label/range name. Grape, colour,
    sweetness and volume differ between the shelf tag and the sheet all the time and
    must not drive the decision.
  - A shared word is only evidence if it is a brand word. Appellations and regions
    (Asti, Chianti, Bordeaux, Dealu Mare), grape names, and generic words
    (Domeniile, Crama, Casa, Vinul) are shared by many unrelated producers.
  - Romanian shelf tags abbreviate hard: "CB. SAUV" is Cabernet Sauvignon,
    "TAM. ROM" is Tamaioasa Romaneasca, "FET. NEAGRA" is Feteasca Neagra.
    Diacritics are dropped inconsistently. Judge on the brand words that remain.
  - Names are often clipped at a crop edge, starting or ending mid-word ("...ANESC").
    Match on what survives if it is distinctive; otherwise choose 0.
  - **Choose 0 whenever the sheet plainly does not contain this brand.** A wrong
    match is worse than none - this column will be trusted for grouping. Many shelf
    products are imports or private label that the sheet never lists.
  - Use "low" confidence freely. It is a signal, not a failure.

Return the structured report when the file is written.`,
  { label: `match:${b}`, phase: 'Match', schema: REPORT }
))

const ok = results.filter(Boolean)
return {
  batches: batches.length,
  done: ok.length,
  matched: ok.reduce((n, r) => n + (r.matched || 0), 0),
  unmatched: ok.reduce((n, r) => n + (r.unmatched || 0), 0),
  notes: ok.map(r => r.notes).filter(n => n && n.trim()).slice(0, 12),
}
