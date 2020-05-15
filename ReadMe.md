# sentland.py

A Python script to generate landscapes matching those in Geoff Crammond's
classic 80's game: The Sentinel (aka The Sentry).

The script generates the landscape data but does not place any objects.

## Requirements

The script requires Python 3.6 or later with the _NumPy_ package installed. You
can install NumPy using:

```
python -m pip install numpy
```

Viewing landscapes with `-v` also requires the _matplotlib_ package, which can
be installed using:

```
python -m pip install matplotlib
```

## Usage

Generate landscape 1234 and save to `1234.bin` (data in row order):
```
python sentland.py 1234
```

Generate landscape 9999 and save to `9999.bin` (data in game memory format):
```
python sentland.py 9999 -m
```

Generate and view landscape 0000:
```
python sentland.py 0 -v
```

Here's a sample view of landscape 0000, shown using matplot:

![Landscape 0000](images/matplot.png)

## Landscape Data

The game stores the landscape as an 32x32 array of vertices, each represented
by a single byte. This gives the 31x31 arrangement of tiles that make up the
landscape.

Each byte holds two nibbles of data. The upper 4 bits hold the vertex height,
and the lower 4 bits a code indicating the shape of the tile. The shape helps
the game quicly know whether a tile is flat, as well as how to draw it, without
needing to look at the neighbouring vertices each time.

If the height value is 12 or above it means there is an object placed on a
flat tile. In that case the low 6 bits is an index into various object tables.
It may be necessary to walk a chain of stacked objects to locate the original
tile height. The map data generated by this script doesn't contain any objects
so these values will not be seen here.

The in-memory layout of the map is not a plain 2D array of values. To match the
original memory layout use the `-m` option when saving data.

## Landscape Generation

All landscapes are generated using the following steps:

### 1) Seed the RNG

The RNG is seeded using the landscape number, as described in the
[sentcode.py](https://github.com/simonowen/sentcode) project.

### 2) Warm the RNG

To ensure good random numbers are provided, 81 values are read from the RNG into
a buffer. Note: these values are not used during the remainder of the process.

### 3) Height scale

Height scaling determines the range between the lowest and highest points on
the landscape. Lower values keep the landscape flatter and easier to navigate.
Landscape 0000 uses a fixed scaling factor of 0x18, but all other landscapes
use a random value massaged into the range 0x0e to 0x24. This will be used in
step 6 below.

### 4) Random fill

The map area is filled with byte values from the RNG, with rows from back to
front, in right to left order. The reverse ordering is due to the original 6502
game code, which loops backwards as long as the index is positive using the
`BPL` instruction.

### 5) Smoothing passes

The random values are smoothed by averaging groups of 4 values and replacing the
first value with this average. This is performed on each row from back to front,
then each column from right to left. This process is repeated a second time.

### 6) Scale and offset

The smoothed values are treated as signed 7-bit values by subtracting each from
0x80. The result is scaled by multiplying it by the height factor from step 3
above, then taking the upper 8 bits of the result to give the height. This is
then offset by 6 to re-centre in the middle of the height range, before being
clamped into the legal range of 1 to 11.

### 7) De-spike passes

To improve the appearance of the map all single vertex spikes and troughs are
removed. This is again performed on rows from back to front, and columns from
right to left. This process is repeated a second time.

De-spiking takes spans of 3 vertices, and checks if vertex 1 and 3 are both
above _or_ both below vertex 2. If they are the height of vertex 2 is set to the
lower of the other two heights.

This completes the final height generation, but not the stored data.

### 8) Shape codes

To simplify use of the map at runtime, the groups of 4 vertices that form each
tile are compared to determine whether it's level and how it will appear when
drawn. This forms a 4-bit code, which is stored (for now) in the upper 4 bits of
each map value.

A code of zero means the tile is level (all vertex heights match). See the
script code for the layouts that generate the other values. Value 8 not used.

### 9) Nibble swap

For some reason the game prefers to work with the height in the upper 4 bits and
the shape in the lower 4 bits. This step simply reverses order of the nibbles.

The original game will continue from this point to place objects, again guided
by values from the RNG.

## Test Data

The '_golden_' sub-directory contains data exported after each generation step
listed above. It was exported from address `0x6100` in the Spectrum version of
the game. Data for landscapes 0000 and 9999 is provided, and compared against
generated data for these levels.

---

Simon Owen  
https://simonowen.com
