## Tips:

- Use blender python code with modal to prevent UI from locking up when running code.

## Blender Depth Map and Object Segmentation Mask:

Can be useful for deep learning. In `View layer properties > Passes > Data` turn on `Mist`, from `viewport shading` dropdown in 3d viewport change `Render Pass` from `Combined` to `Mist`. Further control available under `Word Properties > Mist Pass`.

In the `Render Results` pop up window when `F12` is pressed, change from `Composite` to `View Layer`. Then change `combined` dropdown to mist. In next drop down `Color` will give a depth map and selecting `Z-buffer` will give object segmentation mask. Though this method will not differentiate different object types.
