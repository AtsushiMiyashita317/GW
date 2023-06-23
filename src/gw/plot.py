from typing import List

import torch
from matplotlib import pyplot as plt
from matplotlib import widgets as wg


def calculate_figsize(heights, widths, slider_size):
    height = sum(heights)
    width = sum(widths)
    scale = 4*slider_size[1]/width
    slider_height = 0.5*(slider_size[0]+2)
    image_height = scale*height
    height = image_height + slider_height
    width = scale*width
    slider_height = int(slider_height//scale+1)
    image_height = int(image_height//1+1)
    return height, width, image_height, slider_height

def create_ax_images(fig, heights, widths, slider_height):
    ax_images = []
    height = sum(heights)
    width = sum(widths)
    size = height+slider_height,width
    i = 0
    for h in heights:
        j = 0
        for w in widths:
            ax_images.append(plt.subplot2grid(size,(i,j),h,w,fig))
            j += w
        i += h
    return ax_images

def create_ax_sliders(fig, slider_size, image_height):
    ax_sliders = []
    height = 2*image_height+slider_size[0]+2
    width = slider_size[1]
    size = height,width
    for i in range(slider_size[0]):
        for j in range(slider_size[1]):
            ax_sliders.append(plt.subplot2grid(size,(height+i-slider_size-2,j),1,1,fig))
    return ax_sliders

def init_sliders(ax_sliders, slider_names, **kwargs):
    sliders:List[wg.Slider] = []
    for ax_slider, name in zip(ax_sliders, slider_names):
        sliders.append(wg.Slider(ax_slider, name, **kwargs))
    return sliders 

def demo(heights, widths, slider_size, slider_nemes, slider_args, init_hook, update_hook):
    height, width, image_height, slider_height = calculate_figsize(heights, widths, slider_size)

    fig = plt.figure(figsize=(width,height))
    
    ax_images = create_ax_images(fig, heights, widths, slider_height)
    ax_sliders = create_ax_sliders(fig, slider_size, image_height)

    l_images = init_hook(ax_images)
    sliders = init_sliders(ax_sliders, slider_nemes, **slider_args)

    def update(event):
        update_hook([slider.val for slider in sliders], l_images)
        fig.canvas.draw_idle()

    for i,slider in enumerate(sliders):
        slider.on_changed(update)
        
    ax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    button1 = wg.Button(ax, 'Reset', hovercolor='0.975')

    def reset(event):
        for slider in sliders:
            slider.eventson = False
            slider.reset()
            slider.eventson = True
        
    button1.on_clicked(reset)

    plt.show()