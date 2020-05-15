#!/usr/bin/env python
#
# Landscape generator for The Sentinel (aka The Sentry)
#
# Generates the landscapes from the original game, excluding placed objects.
#
# By Simon Owen https://github.com/simonowen/sentland

import sys
import os.path
import struct
import argparse
import numpy as np

num_landscapes = 0xe000     # includes extended hex landscapes
ull = 0

def get_offset(x, z):
    """Convert x and z coordinates to linear offset into game map data"""
    return ((x & 3) << 8) | ((x & 0x1c) << 3) | z

def get_x_z(offset):
    """Convert linear game map data offset to x and z coordinates"""
    x = ((offset & 0x300) >> 8) | ((offset & 0xe0) >> 3)
    z = (offset & 0x1f)
    return x, z

def at_offset(maparr, offset):
    """Return map entry at given original game map offset"""
    x, z = get_x_z(offset)
    return maparr[z][x]

def wrapped_slice(maparr, entries=0x23, *, x=None, z=None):
    """Return x or z slice from the map, wrapped around at the edges"""
    if z is not None:
        return [maparr[z][x & 0x1f] for x in range(entries)]
    else:
        return [maparr[z & 0x1f][x] for z in range(entries)]

def smooth_slice(arr):
    """Smooth a map slice by averaging neighbouring groups of values"""
    group_size = len(arr) - 0x1f
    return [sum(arr[x:x+group_size]) // group_size for x in range(len(arr) - group_size + 1)]

def smooth_map(maparr, axis):
    """Smooth the map by averaging groups across the given axis"""
    if axis == 'z':
        return [smooth_slice(wrapped_slice(maparr, z=z)) for z in range(0x20)]

    new_maparr = np.empty_like(maparr)
    for x in range(0x20):
        new_maparr[:, x] = smooth_slice(wrapped_slice(maparr, x=x))
    return new_maparr

def despike_midval(arr):
    """Return lowest neighbour if peak or trough, else middle value"""
    if (arr[0] > arr[1] < arr[2]) or (arr[0] < arr[1] > arr[2]):
        return min(arr[0], arr[2])
    else:
        return arr[1]

def despike_slice(arr):
    """Smooth a slice by flattening single vertex peaks and troughs"""
    arr_copy = arr[:]
    for x in reversed(range(1, len(arr_copy) - 1)):
        arr_copy[x] = despike_midval(arr_copy[x-1:x+2])
    return arr_copy[:32]

def despike_map(maparr, axis):
    """De-spike the map in slices across the given axis"""
    if axis == 'z':
        return [despike_slice(wrapped_slice(maparr, z=z)) for z in range(0x20)]

    new_map = np.empty_like(maparr)
    for x in range(0x20):
        new_map[:, x] = despike_slice(wrapped_slice(maparr, x=x))
    return new_map

def scale_and_offset(val, scale=0x18):
    """Scale and offset values to generate vertex heights"""
    mag = val - 0x80            # 7-bit signed range
    mag = mag * scale // 256    # scale and use upper 8 bits
    mag = max(mag + 6, 0)       # centre at 6 and limit minimum
    mag = min(mag + 1, 11)      # raise by 1 and limit maximum
    return mag

def tile_shape(fl, bl, br, fr):
    """Determine tile shape code from 4 vertex heights"""
    if fl == fr:
        if fl == bl:
            if fl == br:
                shape = 0
            elif fl < br:
                shape = 0xa
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
                shape = 0xf
        else:
            shape = 0xc
    elif fl == bl:
        if br == fr:
            if br < bl:
                shape = 0x5
            else:
                shape = 0xd
        elif br == bl:
            if br < fr:
                shape = 0xe
            else:
                shape = 0x7
        else:
            shape = 0x4
    elif br == fr:
        if br == bl:
            if br < fl:
                shape = 0xb
            else:
                shape = 0x2
        else:
            shape = 0x4
    else:
        shape = 0xc

    return shape

def add_tile_shapes(maparr):
    """Add tile shape code to upper 4 bits of each tile"""
    new_maparr = np.copy(maparr)
    for z in reversed(range(0x1f)):
        for x in reversed(range(0x1f)):
            fl = maparr[z+0, x+0] & 0xf
            bl = maparr[z+1, x+0] & 0xf
            br = maparr[z+1, x+1] & 0xf
            fr = maparr[z+0, x+1] & 0xf
            shape = tile_shape(fl, bl, br, fr)
            new_maparr[z][x] = (shape << 4) | (maparr[z][x] & 0xf)
    return new_maparr

def swap_nibbles(maparr):
    return [[((maparr[z][x] & 0xf) << 4) | (maparr[z][x] >> 4)
        for x in range(0x20)]
            for z in range(0x20)]

def seed(landscape_bcd):
    """Seed RNG using landscape number"""
    global ull
    ull = (1 << 16) | landscape_bcd

def rng():
    """Pull next 8-bit value from random number generator"""
    global ull
    for _ in range(8):
    	ull <<= 1
    	ull |= ((ull >> 20) ^ (ull >> 33)) & 1

    return (ull >> 32) & 0xff

def rng_00_16():
    """Random number in range 0 to 0x16"""
    r = rng()
    return (r & 7) + ((r >> 3) & 0xf)

def arr_to_memory(maparr):
    """Convert array data to in-memory format used by game"""
    return bytes([at_offset(maparr, x) for x in range(1024)])

def verify(maparr, landscape_bcd, name):
    """Verify the map data against golden images, if they exist"""
    filename = f'golden/{landscape_bcd:04X}_{name}.bin'
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            if f.read() != arr_to_memory(maparr):
                sys.exit(f'Data mismatch against {filename}')

def generate_landscape(landscape_bcd):
    # Seed RNG using landscape number in BCD.
    seed(landscape_bcd)

    # Read 81 values to warm the RNG.
    [rng() for _ in range(0x51)]

    # Random height scaling (but fixed value for landscape 0000!).
    height_scale = (rng_00_16() + 0x0e) if landscape_bcd else 0x18

    # Fill the map with random values (z from back to front, x from right to left).
    maparr = list(reversed([list(reversed([rng()
        for x in range(0x20)]))
            for z in range(0x20)]))
    verify(maparr, landscape_bcd, 'random')

    # 2 passes of smoothing, each across z-axis then x-axis.
    for _ in range(2):
        maparr = smooth_map(maparr, 'z')
        maparr = smooth_map(maparr, 'x')
    verify(maparr, landscape_bcd, 'smooth3')

    # Scale and offset values to give vertex heights in range 1 to 11.
    maparr = [[scale_and_offset(x, height_scale) for x in z] for z in maparr]
    verify(maparr, landscape_bcd, 'scaled')

    # Two de-spike passes, each across z-axis then x-axis.
    for _ in range(2):
        maparr = despike_map(maparr, 'z')
        maparr = despike_map(maparr, 'x')
    verify(maparr, landscape_bcd, 'despike3')

    # Add shape codes for each tile, to simplify examining the landscape.
    maparr = add_tile_shapes(maparr)
    verify(maparr, landscape_bcd, 'shape')

    # Finally, swap the high and low nibbles in each byte for the final format.
    maparr = swap_nibbles(maparr)
    verify(maparr, landscape_bcd, 'swap')

    return maparr

def view_landscape(maparr):
    """Crude viewing of generated landscape data"""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import LinearLocator
    except:
        sys.exit("Landscape requires matplotlib package:\n  python -m pip install matplotlib")

    X = np.arange(0, 0x20, 1)
    X, Y = np.meshgrid(X, X)
    Z = np.array(maparr) >> 4   # map just height nibble

    flat_colours = ((0.0, 1.0, 0.0), (0.0, 0.62, 0.62)) # light green, dark green
    slope_colours = ((0.6, 0.6, 0.6), (0.7, 0.7, 0.7))  # light grey, dark grey

    colors = np.empty(X.shape, dtype='3f')
    for y in range(len(Y)):
        for x in range(len(X)):
            if maparr[y][x] & 0xf:
                colors[y, x] = slope_colours[(x + y) & 1]
            else:
                colors[y, x] = flat_colours[(x + y) & 1]

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, facecolors=colors, linewidth=0)
    ax.set_zlim(1, 11)
    ax.zaxis.set_major_locator(LinearLocator(6))

    plt.show()

def main(args):
    if args.landscape == None:
        parser.print_help()
    elif args.landscape < 0 or args.landscape >= num_landscapes:
        sys.exit(f'Landscape number must be in range 0000-{num_landscapes-1:04X}')
    else:
        maparr = generate_landscape(args.landscape)

        if args.view:
            view_landscape(maparr)
        else:
            filename = f'{args.landscape:04X}.bin'
            with open(filename, 'wb') as f:
                if args.memory:
                    f.write(arr_to_memory(maparr))
                else:
                    f.write(np.array(maparr, dtype='B').tobytes())

                if not args.quiet:
                    print(f'Wrote landscape {args.landscape:04X} to {filename}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Landscape generator for The Sentinel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('landscape', help='landscape number', type=lambda x: int(x,16), nargs='?')
    parser.add_argument('-v', '--view', help='view landscape in matplot', action='store_true', default=False)
    parser.add_argument('-m', '--memory', help='save data in game memory format', action='store_true', default=False)
    parser.add_argument('-q', '--quiet', help='suppress save message', action='store_true', default=False)
    main(parser.parse_args())
