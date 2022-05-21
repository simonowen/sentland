#!/usr/bin/env python3
#
# Landscape generator for The Sentinel (aka The Sentry)
#
# Generates the landscapes and placed object from the original game.
#
# By Simon Owen https://github.com/simonowen/sentland

import sys
import struct
import os.path
import argparse
import numpy as np  # python -m pip install numpy
from enum import IntEnum

num_landscapes = 0xE000  # includes extended hex landscapes
ull = 0


class ObjType(IntEnum):
    NONE = -1
    ROBOT = 0
    SENTRY = 1
    TREE = 2
    BOULDER = 3
    MEANIE = 4
    SENTINEL = 5
    PEDESTAL = 6


class Object:
    def __init__(self, type, x, y, z):
        self.type = type
        self.x = x
        self.y = y
        self.z = z
        self.rot = None
        self.step = None
        self.timer = None

    def __str__(self):
        """Generate string representation of object"""
        name = str(self.type)[str(self.type).find(".") + 1 :].capitalize()
        rotdeg = None if self.rot is None else f"{int(self.rot * 360 / 256):03}\u00b0"
        rotdir = "" if self.step is None else " \u21ba" if self.step < 0 else " \u21bb"

        s = f"{name+':':9s} x={self.x:02X} y={self.y:02X} z={self.z:02X}"
        if self.rot is not None:
            s += f" rot={self.rot:02X} ({rotdeg}{rotdir})"
        if self.timer is not None:
            s += f" next={self.timer:02X}"
        return s


def get_offset(x, z):
    """Convert x and z coordinates to linear offset into game map data"""
    return ((x & 3) << 8) | ((x & 0x1C) << 3) | z


def get_x_z(offset):
    """Convert linear game map data offset to x and z coordinates"""
    x = ((offset & 0x300) >> 8) | ((offset & 0xE0) >> 3)
    z = offset & 0x1F
    return x, z


def at_offset(maparr, offset):
    """Return map entry at given original game map offset"""
    x, z = get_x_z(offset)
    return maparr[z][x]


def shape_at(x, z, maparr):
    """Return map code at given map location"""
    return maparr[z][x] & 0xF


def height_at(x, z, maparr):
    """Return map height at given location"""
    return maparr[z][x] >> 4


def is_flat(x, z, maparr):
    """Return True if the map location is a flat tile"""
    return shape_at(x, z, maparr) == 0


def objects_at(x, z, objects):
    """Return a list of objects stacked at map location"""
    return [o for o in objects if o.x == x and o.z == z]


def wrapped_slice(maparr, entries=0x23, *, x=None, z=None):
    """Return x or z slice from the map, wrapped around at the edges"""
    if z is not None:
        return [maparr[z][x & 0x1F] for x in range(entries)]
    else:
        return [maparr[z & 0x1F][x] for z in range(entries)]


def smooth_slice(arr):
    """Smooth a map slice by averaging neighbouring groups of values"""
    group_size = len(arr) - 0x1F
    return [
        sum(arr[x : x + group_size]) // group_size
        for x in range(len(arr) - group_size + 1)
    ]


def smooth_map(maparr, axis):
    """Smooth the map by averaging groups across the given axis"""
    if axis == "z":
        return [smooth_slice(wrapped_slice(maparr, z=z)) for z in range(0x20)]

    new_maparr = np.empty_like(maparr)
    for x in range(0x20):
        new_maparr[:, x] = smooth_slice(wrapped_slice(maparr, x=x))
    return new_maparr


def despike_midval(arr):
    """Smooth 3 map vertices, returning a new central vertex height"""
    if arr[1] == arr[2]:
        return arr[1]
    elif arr[1] > arr[2]:
        if arr[1] <= arr[0]:
            return arr[1]
        elif arr[0] < arr[2]:
            return arr[2]
        else:
            return arr[0]
    elif arr[1] >= arr[0]:
        return arr[1]
    elif arr[2] < arr[0]:
        return arr[2]
    else:
        return arr[0]


def despike_slice(arr):
    """Smooth a slice by flattening single vertex peaks and troughs"""
    arr_copy = arr[:]
    for x in reversed(range(0x20)):
        arr_copy[x + 1] = despike_midval(arr_copy[x : x + 3])
    return arr_copy[:32]


def despike_map(maparr, axis):
    """De-spike the map in slices across the given axis"""
    if axis == "z":
        return [despike_slice(wrapped_slice(maparr, z=z)) for z in range(0x20)]

    new_map = np.empty_like(maparr)
    for x in range(0x20):
        new_map[:, x] = despike_slice(wrapped_slice(maparr, x=x))
    return new_map


def scale_and_offset(val, scale=0x18):
    """Scale and offset values to generate vertex heights"""
    mag = val - 0x80  # 7-bit signed range
    mag = mag * scale // 256  # scale and use upper 8 bits
    mag = max(mag + 6, 0)  # centre at 6 and limit minimum
    mag = min(mag + 1, 11)  # raise by 1 and limit maximum
    return mag


def tile_shape(fl, bl, br, fr):
    """Determine tile shape code from 4 vertex heights"""
    if fl == fr:
        if fl == bl:
            if fl == br:
                shape = 0
            elif fl < br:
                shape = 0xA
            else:
                shape = 0x3
        elif br == bl:
            if br < fr:
                shape = 0x1
            else:
                shape = 0x9
        elif br == fr:
            if br < bl:
                shape = 0x6
            else:
                shape = 0xF
        else:
            shape = 0xC
    elif fl == bl:
        if br == fr:
            if br < bl:
                shape = 0x5
            else:
                shape = 0xD
        elif br == bl:
            if br < fr:
                shape = 0xE
            else:
                shape = 0x7
        else:
            shape = 0x4
    elif br == fr:
        if br == bl:
            if br < fl:
                shape = 0xB
            else:
                shape = 0x2
        else:
            shape = 0x4
    else:
        shape = 0xC

    return shape


def add_tile_shapes(maparr):
    """Add tile shape code to upper 4 bits of each tile"""
    new_maparr = np.copy(maparr)
    for z in reversed(range(0x1F)):
        for x in reversed(range(0x1F)):
            fl = maparr[z + 0, x + 0] & 0xF
            bl = maparr[z + 1, x + 0] & 0xF
            br = maparr[z + 1, x + 1] & 0xF
            fr = maparr[z + 0, x + 1] & 0xF
            shape = tile_shape(fl, bl, br, fr)
            new_maparr[z][x] = (shape << 4) | (maparr[z][x] & 0xF)
    return new_maparr


def swap_nibbles(maparr):
    """Swap upper and lower 4 bits in each map byte"""
    return [
        [((maparr[z][x] & 0xF) << 4) | (maparr[z][x] >> 4) for x in range(0x20)]
        for z in range(0x20)
    ]


def seed(landscape_bcd):
    """Seed RNG using landscape number"""
    global ull, rng_usage
    ull = (1 << 16) | landscape_bcd
    rng_usage = 0


def rng():
    """Pull next 8-bit value from random number generator"""
    global ull, rng_usage
    for _ in range(8):
        ull <<= 1
        ull |= ((ull >> 20) ^ (ull >> 33)) & 1

    rng_usage += 1
    return (ull >> 32) & 0xFF


def rng_00_16():
    """Random number in range 0 to 0x16"""
    r = rng()
    return (r & 7) + ((r >> 3) & 0xF)


def arr_to_memory(maparr):
    """Convert array data to in-memory format used by game"""
    return bytes([at_offset(maparr, x) for x in range(1024)])


def verify(maparr, landscape_bcd, name):
    """Verify the map data against golden images, if they exist"""
    filename = f"golden/{landscape_bcd:04X}_{name}.bin"
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            if f.read() != arr_to_memory(maparr):
                sys.exit(f"Data mismatch against {filename}")


def generate_landscape(landscape_bcd):
    """Generate landscape data for given landscape number"""
    # Seed RNG using landscape number in BCD.
    seed(landscape_bcd)

    # Read 81 values to warm the RNG.
    [rng() for _ in range(0x51)]

    # Random height scaling (but fixed value for landscape 0000!).
    height_scale = (rng_00_16() + 0x0E) if landscape_bcd else 0x18

    # Fill the map with random values (z from back to front, x from right to left).
    maparr = list(
        reversed([list(reversed([rng() for x in range(0x20)])) for z in range(0x20)])
    )
    verify(maparr, landscape_bcd, "random")

    # 2 passes of smoothing, each across z-axis then x-axis.
    for _ in range(2):
        maparr = smooth_map(maparr, "z")
        maparr = smooth_map(maparr, "x")
    verify(maparr, landscape_bcd, "smooth3")

    # Scale and offset values to give vertex heights in range 1 to 11.
    maparr = [[scale_and_offset(x, height_scale) for x in z] for z in maparr]
    verify(maparr, landscape_bcd, "scaled")

    # Two de-spike passes, each across z-axis then x-axis.
    for _ in range(2):
        maparr = despike_map(maparr, "z")
        maparr = despike_map(maparr, "x")
    verify(maparr, landscape_bcd, "despike3")

    # Add shape codes for each tile, to simplify examining the landscape.
    maparr = add_tile_shapes(maparr)
    verify(maparr, landscape_bcd, "shape")

    # Finally, swap the high and low nibbles in each byte for the final format.
    maparr = swap_nibbles(maparr)
    verify(maparr, landscape_bcd, "swap")

    return maparr


def view_landscape(maparr):
    """Crude viewing of generated landscape data"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LinearLocator
    except ModuleNotFoundError:
        sys.exit(
            "Landscape requires matplotlib package:\n  python -m pip install matplotlib"
        )

    X = np.arange(0, 0x20, 1)
    X, Y = np.meshgrid(X, X)
    Z = np.array(maparr) >> 4  # map just height nibble

    flat_colours = ((0.0, 1.0, 0.0), (0.0, 0.62, 0.62))  # light green, dark green
    slope_colours = ((0.6, 0.6, 0.6), (0.7, 0.7, 0.7))  # light grey, dark grey

    colors = np.empty(X.shape, dtype="3f")
    for y in range(len(Y)):
        for x in range(len(X)):
            if maparr[y][x] & 0xF:
                colors[y, x] = slope_colours[(x + y) & 1]
            else:
                colors[y, x] = flat_colours[(x + y) & 1]

    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)
    ax.set_zlim(1, 11)
    ax.zaxis.set_major_locator(LinearLocator(6))

    plt.show()


def calc_num_sentries(landscape_bcd):
    """Determine number of sentries on landscape"""
    # Only ever the Sentinel on the first landscape.
    if landscape_bcd == 0x0000:
        return 1

    # Base count uses landscape BCD thousands digit, offset by 2.
    base_sentries = ((landscape_bcd & 0xF000) >> 12) + 2

    while True:
        r = rng()
        # count leading zeros on b6-0 for adjustment size
        adjust = (format(r & 0x7F, "07b") + "1").find("1")
        # b7 determines adjustment sign (note: 1s complement)
        if r & 0x80:
            adjust = ~adjust

        num_sentries = base_sentries + adjust
        if 0 <= num_sentries <= 7:
            break

    # Levels under 100 use tens digit to limit number of sentries.
    max_sentries = (landscape_bcd & 0x00F0) >> 4
    if landscape_bcd >= 0x0100 or max_sentries > 7:
        max_sentries = 7

    # Include Sentinel in sentry count.
    return 1 + min(num_sentries, max_sentries)


def highest_positions(maparr):
    """Find the highest placement positions in 4x4 regions on the map"""
    grid_max = []

    # Scan the map as 64 regions of 4x4 (less one on right/back edges)
    # in z order from front to back and x from left to right.
    for i in range(0x40):
        gridx, gridz = ((i & 7) << 2), ((i & 0x38) >> 1)
        max_height, max_x, max_z = 0, -1, -1

        # Scan each 4x4 region, z from front to back, x from left to right.
        for j in range(0x10):
            x, z = gridx + (j & 3), gridz + (j >> 2)

            # The back and right edges are missing a tile, so skip.
            if x == 0x1F or z == 0x1F:
                continue

            height = height_at(x, z, maparr)
            if is_flat(x, z, maparr) and height >= max_height:
                max_height, max_x, max_z = height, x, z

        grid_max.append([max_height, max_x, max_z])

    return grid_max


def object_at(type, x, y, z):
    """Place object at given position but with random rotation"""
    obj = Object(type, x, y, z)

    # Random rotation, limited to 32 steps, biased by +135 degrees.
    obj.rot = ((rng() & 0xF8) + 0x60) & 0xFF
    return obj


def random_coord():
    """Calculate random map axis coordinate"""
    while True:
        r = rng() & 0x1F
        if r < 0x1F:
            return r


def object_random(type, max_height, objects, maparr):
    """Generate given object at a random unused position below the given height"""
    while True:
        for attempt in range(0xFF):
            x, z = random_coord(), random_coord()
            y = height_at(x, z, maparr)

            if (
                is_flat(x, z, maparr)
                and not objects_at(x, z, objects)
                and y < max_height
            ):
                return object_at(type, x, y, z)

        max_height += 1
        if max_height >= 0xC:
            return None


def place_sentries(landscape_bcd, maparr):
    """Place Sentinel and appropriate sentry count for given landscape"""
    objects = []
    highest = highest_positions(maparr)
    max_height = max([x[0] for x in highest])

    num_sentries = calc_num_sentries(landscape_bcd)
    for _ in range(num_sentries):
        while True:
            # Filter for high positions at the current height limit.
            height_indices = [i for i, x in enumerate(highest) if x[0] == max_height]
            if height_indices:
                break

            # No locations so try 1 level down, stopping at zero.
            max_height -= 1
            if max_height == 0:
                return objects, max_height

        # Results are in reverse order due to backwards 6502 iteration loop.
        height_indices = list(reversed(height_indices))

        # Mask above number of entries to limit random scope.
        idx_mask = 0xFF >> format(len(height_indices), "08b").find("1")
        while True:
            idx = rng() & idx_mask
            if idx < len(height_indices):
                break

        idx_grid = height_indices[idx]
        y, x, z = highest[idx_grid]

        # Invalidate the selected and surrounding locations by setting zero height.
        for offset in [-9, -8, -7, -1, 0, 1, 7, 8, 9]:
            idx_clear = idx_grid + offset
            if idx_clear >= 0 and idx_clear < len(highest):
                highest[idx_clear][0] = 0

        if not objects:
            pedestal = object_at(ObjType.PEDESTAL, x, y, z)
            pedestal.rot = 0
            objects.append(pedestal)
            objects.append(object_at(ObjType.SENTINEL, x, y + 1, z))
        else:
            objects.append(object_at(ObjType.SENTRY, x, y, z))

        # Generate rotation step/direction and timer delay from RNG.
        r = rng()
        objects[-1].step = -20 if (r & 1) else +20
        objects[-1].timer = ((r >> 1) & 0x1F) | 5

    return objects, max_height


def place_player(landscape_bcd, max_height, objects, maparr):
    """Place player robot on the landscape"""

    # The player position is fixed on landscape 0000.
    if landscape_bcd == 0x0000:
        x, z = 0x08, 0x11
        player = object_at(ObjType.ROBOT, x, height_at(x, z, maparr), z)
    else:
        # Player is never placed above height 6.
        max_player_height = min(max_height, 6)
        player = object_random(ObjType.ROBOT, max_player_height, objects, maparr)

    objects.append(player)
    return objects, max_height


def place_trees(max_height, objects, maparr):
    """Place the appropriate number of trees for the sentry count"""

    # Count the placed Sentinel and sentries.
    num_sents = len(
        [o for o in objects if o.type in [ObjType.SENTINEL, ObjType.SENTRY]]
    )

    r = rng()
    max_trees = 48 - (3 * num_sents)
    num_trees = (r & 7) + ((r >> 3) & 0xF) + 10
    num_trees = min(num_trees, max_trees)

    for _ in range(num_trees):
        tree = object_random(ObjType.TREE, max_height, objects, maparr)
        objects.append(tree)

    return objects, max_height


def main(args):
    if args.landscape is None:
        parser.print_help()
    elif args.landscape < 0 or args.landscape >= num_landscapes:
        sys.exit(f"Landscape number must be in range 0000-{num_landscapes-1:04X}")
    else:
        land = args.landscape
        maparr = generate_landscape(land)
        objects, max_height = place_sentries(land, maparr)
        objects, max_height = place_player(land, max_height, objects, maparr)
        objects, max_height = place_trees(max_height, objects, maparr)

        # Sanity check the RNG usage against values from the original code.
        with open("golden/iterations.bin", "rb") as f:
            iterations = struct.unpack(f"<{num_landscapes}h", f.read())
            if iterations[land] != rng_usage:
                sys.exit(f"RNG mismatch: {rng_usage} != {iterations[land]}")

        if args.view:
            view_landscape(maparr)
        else:
            filename = f"{land:04X}.bin"
            with open(filename, "wb") as f:
                if args.memory:
                    f.write(arr_to_memory(maparr))
                else:
                    f.write(np.array(maparr, dtype="B").tobytes())

                if not args.quiet:
                    print(f"Wrote landscape {land:04X} to {filename}")
                    print("Objects:")
                    for o in objects:
                        print(f"  {o}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Landscape generator for The Sentinel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "landscape", help="landscape number", type=lambda x: int(x, 16), nargs="?"
    )
    parser.add_argument(
        "-v",
        "--view",
        help="view landscape in matplot",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-m",
        "--memory",
        help="save data in game memory format",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-q",
        "--quiet",
        help="suppress save message",
        action="store_true",
        default=False,
    )
    main(parser.parse_args())
