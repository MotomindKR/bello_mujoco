# Bello face app icon

`bello-face.png` is the source artwork for the desktop icon. It depicts Mini's
white head shell, vermilion face panel, and dark two-camera visor. Created with
the built-in imagegen tool on 2026-09-29; no third-party stock art is included.
Transparent corners are preserved. Packaging creates the macOS ICNS and its
16-1024 pixel representations under ignored `out/`; Linux uses the PNG.
The DEB and RPM packages install that PNG as the desktop launcher icon.
`package.json` supplies `bello-mini-console.desktop` to Electron so Linux can
match the running app to its launcher. Release CI checks that the extracted DEB
and RPM icons are byte-for-byte identical to this source and that both desktop
entries match the source template.

Generation prompt:

> Create one finished desktop application icon for Bello Mini, a robotics
> engineering console by Motomind. Square 1024x1024 composition. A centered,
> nearly front-facing simplified robot HEAD ONLY: tall softly rounded
> rectangular ivory-white outer shell, inset bright vermilion red-orange
> (#E82E01) face panel, a short horizontal dark charcoal rectangular camera
> visor in the upper middle with exactly TWO small circular camera lenses
> spaced apart. This is a physical robot sensor face, no mouth, no antenna,
> no ears, no cartoon eyes, no letters. Head slightly taller than wide, clean
> precision-molded shell with subtle bevel, generous clear silhouette. Calm
> refined industrial product icon, restrained shallow 3D dimensionality, soft
> studio light, sharp clean edges, no busy texture. Place the head large and
> centered on a deep charcoal (#191f28) macOS-style rounded-square tile;
> transparent outside the tile corners. Tile occupies about 90% of canvas,
> head about 62% width and 75% height of tile. Keep recognizable at 32px.
> No text, no watermark, no robot body, no UI, no additional objects, one icon only.
