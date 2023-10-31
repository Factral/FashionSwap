# ${\textsf{\color{pink}FashionSwap: Image-Based Clothing Cropping for Style Transformation}}

## Overview
FashionSwap is an innovative AI project that focuses on the transformation of clothing styles. Using a dataset of individuals wearing different clothing items, the system segments or detects the clothing item, crops it, transfers its style through a GAN, and finally fits it back onto the individual. This way, we can visualize how a transformed clothing style would look on a person.

## Key Features
1. **Clothing Segmentation & Cropping** - Extract clothing from the background.
2. **Style Transformation using GAN** - Transfer the clothing style to generate a new one.
3. **Fitting the Transformed Clothing** - Visualize how the new clothing style fits on a person.

## Workflow
1. Image Input: Import an image of a person wearing a clothing item.
2. Segmentation: Detect and segment the clothing item from the individual.
3. Cropping: Crop the segmented clothing item.
4. Style Transformation: Use a GAN to transform the clothing's style.
5. Visualization: Place the transformed clothing back onto the original image to see the outcome.

