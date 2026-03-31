# Guidelines-Docker-GVGAI-RLbaselines.md
Step-by-step instructions written by [Hao Tong](https://github.com/HawkTom).

## 1. Install docker 

## 2. Download required packages

- Anaconda [link](https://www.anaconda.com/distribution/)  [清华镜像 (TsingHua Mirror)](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)(2019年10月之后发布的) The anaconda published after 2019.10 is recommended. 
- Java [link](https://www.oracle.com/technetwork/java/javase/downloads/java-archive-javase9-3934878.html): jdk-9 is recommended 

**Only For GPU users**

If you have GPU in your machine, you can use GPU to speed up the learning process. Before doing, you will need to:

* install NAVIDA driver on your machine

* install the docker on your machine

* install NAVIDA docker (It is simply a plugin to Docker to support GPU using in container)

The above three step can be refer to the existing blog by Trung Tran: [Installing NVIDIA Docker On Ubuntu 16.04](https://chunml.github.io/ChunML.github.io/project/Installing-NVIDIA-Docker-On-Ubuntu-16.04/)

## 3. Dockerfile  (modified from the rl_baseline repository)

### CPU only ([download Dockerfile](https://github.com/SUSTechGameAI/GVGAI_GYM/blob/master/docker/Dockerfile.cpu))

```dockerfile
FROM ubuntu:16.04

# Download jdk-9 from  https://www.oracle.com/technetwork/java/javase/downloads/java-archive-javase9-3934878.html
# You can replace ``jdk-9.tar.gz'' by the one that you have downloaded
ADD jdk-9.tar.gz /usr/local

RUN apt-get -y update && apt-get -y install vim git libopenmpi-dev zlib1g-dev cmake libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1

WORKDIR /app

# Download Anaconda from https://www.anaconda.com/distribution/ 
# You can replace ``anaconda.sh'' by the installation script that you have downloaded, e.g., ``Anaconda3-5.3.1-Linux-x86_64.sh''
COPY anaconda.sh /app
RUN bash anaconda.sh -b -p /opt/conda && \
    rm -rf anaconda.sh


ENV JAVA_HOME=/usr/local/jdk-9.0.4
ENV CLASSPATH=.:$JAVA_HOME/lib:$CLASSPATH
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/SUSTechGameAI/GVGAI_GYM.git

# The rl-baseline required ternsorflow version to be from 1.8.0 to 1.14.0
RUN pip install -e GVGAI_GYM/ && \
    pip install codacy-coverage && \
    pip install mpi4py && \
    pip install tensorflow==1.14.0 && \
    pip install cloudpickle && \
    pip install stable-baselines[mpi]

# Install pytorch cpu version by uncommenting the following line   
# RUN pip install torch==1.3.0+cpu torchvision==0.4.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

CMD /bin/bash
```

### GPU version ([download Dockerfile](https://github.com/SUSTechGameAI/GVGAI_GYM/blob/master/docker/Dockerfile.gpu))
- In the dockerfile, use `FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04` instead of `From Ubuntu:16.04`. 

- The version of TensorFlow and Pytorch should be GPU version.

- When run container from image, add `--runtime=nvdia ` in the command. 

  E.g. `docker run --runtime=nvidia -v $PWD:/home --rm -it baseline_gpu /bin/bash`

- Use `nvidia-smi` to find out the status of GPU. 


```dockerfile
FROM nvidia/cuda:9.0-cudnn7-runtime-ubuntu16.04

# Download jdk-9 from  https://www.oracle.com/technetwork/java/javase/downloads/java-archive-javase9-3934878.html
# You can replace ``jdk-9.tar.gz'' by the one that you have downloaded
ADD jdk-9.tar.gz /usr/local

RUN apt-get -y update && apt-get -y install vim git libopenmpi-dev zlib1g-dev cmake libglib2.0-0 libsm6 libxext6 libfontconfig1 libxrender1

WORKDIR /app

# Download Anaconda from https://www.anaconda.com/distribution/ 
# You can replace ``anaconda.sh'' by the installation script that you have downloaded, e.g., ``Anaconda3-5.3.1-Linux-x86_64.sh''
COPY anaconda.sh /app
RUN bash anaconda.sh -b -p /opt/conda && \
    rm -rf anaconda.sh


ENV JAVA_HOME=/usr/local/jdk-9.0.4
ENV CLASSPATH=.:$JAVA_HOME/lib:$CLASSPATH
ENV PATH=$JAVA_HOME/bin:$PATH
ENV PATH=/opt/conda/bin:$PATH

RUN git clone https://github.com/SUSTechGameAI/GVGAI_GYM.git

# The rl-baseline required ternsorflow version to be from 1.8.0 to 1.14.0
RUN pip install -e GVGAI_GYM/ && \
    pip install codacy-coverage && \
    pip install mpi4py && \
    pip install tensorflow-gpu==1.14 && \
    pip install cloudpickle && \
    pip install stable-baselines[mpi]

# Install pytorch cpu version by uncommenting the following line   
# RUN conda install pytorch==1.1.0 torchvision==0.3.0 cudatoolkit==9.0 -c pytorch

CMD /bin/bash
```

*Remark: anaconda.sh jdk.tar.gz and Dockerfile should be in the same folder*!


## 4. Build your image
Now you should have the following files in the same repository:
* jdk-9.tar.gz
* anaconda.sh
* Dockfile

Try the follwing command line in the repository:
```
docker build . -t <image_name>
```

## 5. Testing the installation

   - Use `docker ps` to check whether your image exist or not.
   - Tests the environment by using pytest runner:
     - CPU version: 
  
        `docker run -w /app/GVGAI_GYM --rm <image_name> pytest`

     - GPU version: 
        
        `docker run --runtime=nvidia -w /app/GVGAI_GYM --rm <image_name> pytest`


## 6. Run container from image
**CPU only**
```
docker run -v $PWD:/home -w /home --name <container_name> -it <image_name> /bin/bash
```
**For GPU users**
```
docker run --runtime=nvidia -v $PWD:/home -w /home --name <container_name> -it <image_name> /bin/bash
```

*ps: The `-v` flag mounts the current working directory into the container.*

## Other commands

* Exit a running container: `CTRL + D`

* Load an existing container:
```
docker start <container_name>
```

* Run your docker in the background
add `-d` before `-it` 

* Enter into a container running in the background
```
docker exec -it <contanier_name> /bin/sh
```

**More docker commands refer to the [official tutorial](https://docs.docker.com/engine/reference/commandline/cli/)**

### References

1. [G. Brockman, V. Cheung, L. Pettersson, J. Schneider, J. Schulman, J. Tang, and W. Zaremba, “Openai gym,” 2016.](https://github.com/openai/gym)
2. [A. Hill, A. Raffin, M. Ernestus, A. Gleave, A. Kanervisto, R. Traore, P. Dhariwal, C. Hesse, O. Klimov, A. Nichol, M. Plappert, A. Radford, J. Schulman, S. Sidor, and Y. Wu, “Stable baselines,” https://github.com/hill-a/stable-baselines, 2018.](https://github.com/hill-a/stable-baselines)
3. [R. R. Torrado, P. Bontrager, J. Togelius, J. Liu, and D. Perez-Liebana, “Deep reinforcement learning for general video game ai,” in Computational Intelligence and Games (CIG), 2018 IEEE Conference on. IEEE,
   2018.](https://github.com/rubenrtorrado/GVGAI_GYM)





