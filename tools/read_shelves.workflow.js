export const meta = {
  name: 'read-shelves',
  description: 'Read prepared shelf crops with one agent per photo, writing an answer file per band',
  whenToUse: 'After `wine-ocr prep`. Pass args: {briefs: [{brief, photo, store, bands, answers}], label}',
  phases: [
    { title: 'Read', detail: 'one agent per photo: look at each shelf crop, write its answer JSON' },
  ],
}

// The caller builds the work list from the prep manifest, because a workflow
// script has no filesystem of its own. Two shapes are accepted: full entries
// with a `brief` path, or — much more compact for a run of hundreds — a `work`
// directory plus entries carrying just the photo's sha stem.
const work = (args && args.work) || null
// Three shapes are accepted, in increasing compactness. Full entries carrying a
// `brief` path; entries carrying just the photo's sha stem; or — for a run of
// hundreds, where the arg list itself becomes the expensive part — a flat array
// of "sha" or "sha|Photo name" strings under `shas`.
const raw = (args && args.shas)
  ? args.shas.map(s => {
      const [sha, photo] = String(s).split('|')
      return { sha, photo: photo || sha }
    })
  : ((args && args.briefs) || [])
const briefs = raw.map(b => ({
  ...b,
  brief: b.brief || `${work}/briefs/${b.sha}.md`,
  photo: b.photo || b.sha,
  store: b.store || '(named in the brief)',
  bands: b.bands || 0,
}))
const label = (args && args.label) || 'run'

if (!briefs.length) {
  log('nothing to read — no briefs supplied')
  return { read: 0, failed: 0 }
}

log(`${label}: reading ${briefs.length} photo(s)`)

phase('Read')

const REPORT = {
  type: 'object',
  additionalProperties: false,
  required: ['written', 'skipped', 'notes'],
  properties: {
    written: {
      type: 'array',
      items: { type: 'string' },
      description: 'Absolute paths of the answer files actually written.',
    },
    skipped: {
      type: 'array',
      items: { type: 'string' },
      description: 'Answer paths not written, each with a short reason.',
    },
    notes: {
      type: 'string',
      description: 'Anything about this photo a human should know. Empty string if nothing.',
    },
  },
}

const results = await pipeline(
  briefs,
  (b, _item, i) => agent(
    `Follow the instructions in this file exactly:

  ${b.brief}

It is a self-contained brief for one shop photograph (${b.photo}).
It names every shelf crop to read. For each crop the brief gives you an image path and
an answer path.

Do this for every crop, in order:
  1. Read the image with the Read tool.
  2. Work out the wines and prices per the brief's standing instructions.
  3. Write the JSON object to that crop's answer path with the Write tool.

Rules:
  - Write ONLY the answer files the brief names. Do not create, edit or delete anything else.
  - Each answer file must contain exactly one JSON object matching the schema in the brief —
    no markdown fence, no commentary, no wrapper key.
  - Every field in the schema must be present; use null where you cannot determine a value.
  - Write an answer for every crop, including ones that hold no wine (empty "wines" list,
    with the reason in "notes").
  - Do not run any command that touches the photo library.

On effort. Work from the crop you are given and the transcript beside it. The crop was
cut and sized for this job, so re-cropping or magnifying it yourself is usually wasted
time — and there are hundreds of photos behind this one. Reach for a shell command only
when a specific price is actually in dispute: the transcript says its own tag's
arithmetic disagrees, or the digits plainly differ from the image. Then check that one
price and move on. Do not open a browser.

Where something is unresolvable, the honest answer is a null and a low confidence, not
another five minutes of looking. A thin record with a solid price is useful; a perfect
record that arrives tomorrow is not.

Then return the structured report. Keep "notes" to what a human could act on — a
systematic misread, a shelf the transcript missed, a pairing you are unsure of. It is a
handover note, not a write-up.`,
    { label: `read:${b.photo}`, phase: 'Read', schema: REPORT }
  )
)

const ok = results.filter(Boolean)
const written = ok.reduce((n, r) => n + (r.written || []).length, 0)
const skipped = ok.reduce((n, r) => n + (r.skipped || []).length, 0)
const dead = results.length - ok.length

log(`${label}: ${written} answers written, ${skipped} skipped, ${dead} agent(s) failed`)

return {
  photos: briefs.length,
  written,
  skipped,
  failed: dead,
  notes: ok.map(r => r.notes).filter(n => n && n.trim()),
  skippedDetail: ok.flatMap(r => r.skipped || []),
}
