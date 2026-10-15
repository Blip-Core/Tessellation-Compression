# System Requirements
- **[Python](https://python.org/downloads/release/python-31117)** $>=$ `3.11`
- **[opencv-python](https://pypi.org/project/opencv-python/)** $>=$ `4.0.0.21`
- **[shapely](https://pypi.org/project/shapely/)** $>=$ `2.2.0`
- **[numpy](https://pypi.org/project/numpy/)** $>=$ `2.0.0`
> *Note the script will automatically ensure all packages are installed and of a compatible version.*

# Functions and their Usage
## Table of Functions
#### [`main()`](#main-jump-to-top) - Main Program
#### [`enhance_source_contrast()`](#enhance_source_contrast-jump-to-top) - Contrast Enhancement

## **`main()`** <sup>[Jump to top](#functions-and-their-usage)</sup>

> `main()` is the main portion of the program, and is how you normally interact with the script.
> <br>***Note:** Other functions explained may be extremely similar/identical to the explanation for main()'s variables, as they are directly passed.*

```python
from main import main

main(
    image_path = "Shade_Test.png",
    n_colors = 4,
    auto_contrast = True,
    clip_percent = (1.0, 99.0),
    clahe_strength = 0.30,
    min_shade_gap = 38.0,
    anchor_endpoints = True,
    initial_scale = 2.0,
    scale_step = 1.5,
    max_scale = 0.9,
    min_improvement = 0.05,
    match_tolerance = 14.0,
    dither_mode = "vertical hatch",
    dither_cell_size = 1,
    show_live_preview = True,
)
```

### `image_path` (*Default is `"Shade_Test.png"`*)
Specifies where the script should look for the target image.

> *`image_path` must be a [string](https://docs.python.org/3/builtins/stdtypes.html#text-sequence-type-str) that resolves the [Path](https://docs.python.org/3/library/os.path.html) to the image.*
```python
from main import main

main(
    image_path = "Shade_Test.png"
)
```

### `n_colors` (*Default is `4`*)
Controls the number of shades the script should be looking for.

> *`n_colors` must be an [integer](https://docs.python.org/3/builtins/stdtypes.html#numeric-types-int-float-complex) and can not be smaller than `2`.*

```python
from main import main

main(
    n_colors = 4
)
```

### `auto_contrast` (*Default is `True`*)
Controls if the auto_contrast is enabled or not.

> *`auto_contrast` must be an [boolean](https://docs.python.org/3/builtins/stdtypes.html#boolean-operations-and-or-not).*

```python
from main import main

main(
    auto_contrast = True
)
```

### `clip_percent` (*Default is `(1.0, 99.0)`*)
Controls the amount the auto contrast clips the colors.

> *`clip_percent` must be a [tuple] consisting of two [floats], of which neither are lower than zero, and where the second value is larger than the first.*<br>
> *[`auto_contrast`](#auto_contrast-default-is-true) must be set to `True`.*

```python
from main import main

main(
    auto_contrast = True,
    clip_percent = (1.0, 99.0)
)
```

### `clahe_strength` (*Default is `0.30`*)
Controls the strength of the CLAHE effect.

> *`clahe_strength` must be a [float], and must be larger than zero.*<br>
> *[`auto_contrast`](#auto_contrast-default-is-true) must be set to `True`.*

```python
from main import main

main(
    auto_contrast = True,
    clahe_strength = 0.30
)
```

### `min_shade_gap` (*Default is `38.0*)
Defines the minimum gap between two different shades from the image to ensure distinct colors.

> *`min_shade_gap` must be a [float] and must be greater than zero.*

```python
from main import main

main(
    min_shade_gap = 38.0
)
```

### `anchor_endpoints` (*Default is `True`*)
Determines wheter absolute black and absolute white are forcibly anchored as the darkest and lightest shades in the extracted palette.

> *`anchor_endpoints` must be a [boolean].*

```python
from main import main

main(
    anchor_endpoints = True
)
```

### `initial_scale` (*Default is `2.0`*)
Sets the starting resolution scale factor for the rendering pipeline.

> *`initial_scale` must be a [float] greater than zero.*

```python
from main import main

main(
    initial_scale = 2.0
)
```

### `scale_step` (*Default is `1.5`*)
The multiplier used to increase the image resolution at each iteration during the progressive scaling phase.

> *`scale_step` must be a [float] greater than `1.0`.*

```python
from main import main

main(
    scale_step = 1.5
)
```

### `max_scale` (*Default is `0.9`*)
Defines maximum scale limit for the image.

> *`max_scale` must be a [float] greater than zero.*

```python
from main import main

main(
    max_scale = 0.9
)
```

### `min_improvement` (*Default is `0.05`*)
Minimum required geometric accuracy improvement between scaling steps to justify continuing to run.

> *`min_improvement` must be a [float].*

```python
from main import main

main(
    min_improvement = 0.05
)
```

### `match_tolerance` (*Default is `14.0`*)
Acceptable variance in pixel intensity when calculating geometric accuracy of generated vectors against the original quantized image.

> *`match_tolerance` must be a [float] greater than zero.*

```python
from main import main

main(
    match_tolerance = 14.0
)
```

### `dither_mode` (*Default is `"vertical hatch"`*)
Defines specific pattern applied during dithering phase to mid-band pixels.

> *`dither_mode` must be a [string].*

> The acceptable inputs are `"bayer"`, `"hatch"`, and `"vertical hatch"`.

```python
from main import main

main(
    dither_mode = "vertical hatch"
)
```

### `dither_cell_size` (*Default is `1`*)
Controls scale and spacing of applied dithering pattern matrix.

> *`dither_cell_size` must be an [integer] greater than zero.*

```python
from main import main

main(
    dither_cell_size = 1
)
```

### `show_live_preview` (*Default is `True`*)
Toggles OpenCV window that shows progress of steps and finalized vector image.

> *`show_live_preview` must be a [boolean].*

```python
from main import main

main(
    show_live_preview = True
)
```

## **`enhance_source_contrast()`** <sup>[Jump to top](#functions-and-their-usage)</sup>
> Normalizes the dynamic range via percentile clipping and blends in local contrast enhancement ([CLAHE]()). This results in mid-tone structures standing out cleanly.

```python
from main import enhance_source_contrast

enhance_source_contrast(
    gray_img = base_image,
    clip_percent = (1.0, 99.0),
    clahe_strength = 0.30
)
```

### `gray_img`
Specifies the input image array that the contrast enhancement is applied to.

> *`gray_img` must be a two-dimensional [numpy.ndarray](https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html) representing a grayscale image.

```python
from main import enhance_source_contrast

enhance_source_contrast(
    gray_img = base_image
)
```

More will be added soon.