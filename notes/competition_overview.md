# Competition overview: task, input, and metadata

An introduction to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026):
what is being predicted, what the images look like, and what metadata comes with them.
For analysis results see [findings.md](findings.md); for prior work see [literature.md](literature.md).

## Examples

A typical image with 11 filaments. From left to right: raw input, every filament colored as its own instance, a 512 px
zoom without labels, and the same zoom with labels (L/R = chirality, U = unidentifiable).

![Example image with colored filament instances](figures/20111116063134Lh_instances.png)

The same zoomed region drawn independently by its 3 annotators:

![Three annotators' masks for the same region](figures/20111116063134Lh_annotators.png)

The busiest training image (26 filaments, many of them small and faint):

![Busiest image with 26 filament instances](figures/20140430071914Th_instances.png)

Other images can be drawn with `python scripts/show_example.py --stem <name>` (or `--random 5`). Output goes to
`outputs/examples/`.

## What the competition is

**The science.** Filaments are clouds of cooler, denser gas held above the Sun's surface by magnetic fields. In
H-alpha light (a deep-red wavelength emitted by hydrogen) they show up as dark, thread-like streaks against the bright
disk. When one becomes unstable it can erupt into a coronal mass ejection, which affects space weather on Earth, so
solar physicists want filaments detected and outlined automatically.

**The task.** Given a full-disk image of the Sun, output one mask per filament. This is *instance* segmentation: you
don't just mark "filament pixels", you say which pixels belong to which filament. In the first figure each color is a
separate instance, and each counts separately in the score.

**The scoring (Panoptic Quality).** For every annotator's labelling of an image:
- A predicted filament counts as a hit if it overlaps a true filament with IoU above 0.5. IoU is the shared area
  divided by the combined area.
- The score is roughly the average IoU of the hits, reduced for every missed filament and every false filament:
  PQ = Σ IoU(hits) / (|hits| + 0.5·|false| + 0.5·|missed|).
- Splitting one filament into two pieces usually turns one hit into two misses. That's why fragmentation and merging
  hurt so much.
- On images with several annotators, the same prediction is scored separately against each annotator's set.

## The input

- **887 JPEG images** at 2048×2048, single-channel grayscale: 707 for training and 180 for testing, without labels.
- They come from **GONG** (Global Oscillation Network Group), a network of 6 identical telescopes spread around the
  world so the Sun is always observed. The image's filename tells you which one took it.
- The images span 2011–2022, so they cover both active and quiet phases of the solar cycle.
- What the images look like:
  - The bright disk is about 1,900 px across; outside it is black.
  - The thin bright ring at the edge is the limb, with bright prominences sticking out (these are filaments seen
    side-on, but they aren't labelled).
  - The surface has a grainy "fibril" texture.
  - Bright patches are active regions around sunspots.
  - The dark streaks are the filaments.
- Filaments are only about 12% darker than the surface around them and typically about 8 px wide. In the zoom panels
  you can see how faint the smaller ones are against the texture.

## The metadata

### The filename

`YYYYMMDDHHMMSS` + site letter + `h`. For example, `20111116063134Lh` was taken on 16 Nov 2011 at 06:31:34 UTC at
Learmonth.

| Letter | Site |
|---|---|
| B | Big Bear |
| C | Cerro Tololo |
| L | Learmonth |
| M | Mauna Loa (the blurry one) |
| T | El Teide |
| U | Udaipur |

### The annotation file

`train/MAGFiLO_1.0_Annotations_kaggle2026_train.json` is in COCO format and covers the training images only. Its
three relevant parts:

**`images`:** one entry per *annotator-image*, so 1,154 entries for 707 files. The `id` looks like
`050103-20111116063134Lh`: the first 4 digits are the annotation batch and the last 2 are the annotator within it
(01–03). Each entry also has the size, the capture time and a URL to the original GONG file.

**`annotations`:** 8,199 filaments, each with:

| Field | What it is | Used by the competition? |
|---|---|---|
| `segmentation` | the filament outline as a polygon, i.e. **the training target** | yes |
| `image_id` | which annotator-image it belongs to | yes (for grouping) |
| `area`, `bbox` | size and bounding box, derived from the outline | derived |
| `category_id` | chirality: Left, Right or Unidentifiable (the L/R/U tags in the figures) | no |
| `spine` | a hand-drawn centre line along the filament | no; possibly forbidden for training (to ask the organizers) |
| `iscrowd`, `id` | COCO bookkeeping | no |

**`categories`:** the 4 chirality classes. "Ambiguous" was relabelled during review and never appears.

**Chirality** is the handedness of the filament's magnetic field. It shows in the direction its side "barbs" point, and
it's why annotators were careful about barbs on L/R filaments but drew U filaments as smooth shapes. The competition
ignores the class, but it explains why masks look so different: compare the smooth blue U blob with the frilly green L
filament at the bottom of the busiest image.

### Not provided

No labels for the test images, no exposure or quality metadata, and no magnetograms (magnetic-field maps). Everything
else has to come from the pixels.
