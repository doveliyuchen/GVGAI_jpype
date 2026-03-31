## Visualization in GYM and GVGAI_GYM

### Method 1 (.png -> .gif)

**1. Save each step as an image**

```python
def show_state(env, step, name, info):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title("%s | Step: %d %s" % (name, step, info))
    plt.axis("off")
    path = 'imgs/' + name + str(step) +'.png'
    plt.savefig(path)
```

*Use this function after every step*

**2. Convert all images into .gif image**

```python
import imageio
 
def create_gif(image_list, gif_name): 
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))
    # Save them as frames into a gif 
    imageio.mimsave(gif_name, frames, 'GIF', duration = 0.1)
    return

def main():
    image_list = []
    for i in range(186):
        name = "imgs/game" + str(i) + ".png"
        image_list.append(name)
    gif_name = 'created_gif.gif'
    create_gif(image_list, gif_name)
 
if __name__ == "__main__":
    main()
```

### Method 2 (env -> .gif)

```python
class show_state_gif():
    def __init__(self):
        self.frames = []
    def __call__(self, env):
        self.frames.append(env.render(mode='rgb_array'))

    def save(self, game_name):
        gif_name = game_name + '.gif'
        imageio.mimsave(gif_name, self.frames, 'GIF', duration = 0.1)
```

```python
img = show_state_gif() # initilization
img() # new frame
img.save() # save .gif
```

