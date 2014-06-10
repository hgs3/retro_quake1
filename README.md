# Quake 1 Retro Mod

This is a Quake 1 post-processing effect which intends to emulate lower resolution and reduced color palette.

This mod is for those who want a more "authentic" retro experience.

See high resolution [before](https://raw.githubusercontent.com/zippers/retro_quake1/screenshots/high_res_h.gif) and [after](https://raw.githubusercontent.com/zippers/retro_quake1/screenshots/retro_look_h.gif).

#### Original

![before](https://raw.githubusercontent.com/zippers/retro_quake1/screenshots/high_ogre.gif)

[See high resolution version](https://raw.githubusercontent.com/zippers/retro_quake1/screenshots/high_ogre_h.gif).

#### Retro

![before](https://raw.githubusercontent.com/zippers/retro_quake1/screenshots/retro_ogre.gif)

[See high resolution version](https://raw.githubusercontent.com/zippers/retro_quake1/screenshots/retro_ogre_h.gif).

### Install

1. Obtain a retail version of Quake 1.
2. Download the [darkplaces engine](http://icculus.org/twilight/darkplaces/files/darkplacesengine20140513.zip).
3. Copy the Id1 directory from the retail Quake folder into the extracted darkplaces engine directory.
4. Place the contents of this repos id1 into the Id1 directory.
5. Start darkplaces and enjoy.

### Configuration

To change the resolution and color pallete, open the Quake console from within the game using the ```~``` (tilda) key.

Enter the following command with whatever ```c``` (color range per-channel), ```w``` (emulated width), and ```h``` (emulated height) values you want.

```r_glsl_postprocess_uservec1 "c w h 0"```

So, for example, if you want to emulate 320x240 resolution with 25 colors per-channel, you would enter:

```r_glsl_postprocess_uservec1 "25 320 240 0"```

You can toggle the effect at any time by entering: ```r_glsl_postprocess 1``` to enable and ```r_glsl_postprocess 0``` to disable.


