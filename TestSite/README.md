# Test clips

The clips `lab.html` expects are not in the repository: they are large and they
are re-encodable from public sources. Both are Blender Foundation films under
CC-BY.

- **Big Buck Bunny** — https://archive.org/download/BigBuckBunny_124/Content/big_buck_bunny_720p_surround.mp4
- **Sintel trailer** — https://durian.blender.org/download/

Encode them down to the sizes the lab lists, for example:

```bash
ffmpeg -i big_buck_bunny_720p_surround.mp4 -t 180 -vf scale=640:360 \
  -b:v 450k -c:v libx264 -an TestSite/bbb-360p-450k-3min.mp4
```

The point of the set is a spread of bitrates from roughly 200 kbps to 1 Mbps at
240p through 720p, which is the range Lucid is built for.
