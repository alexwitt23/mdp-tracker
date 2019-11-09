xhost +"local:docker@"

sudo docker run --runtime=nvidia -ti --net=host --ipc=host -e DISPLAY=$DISPLAY \
-v /tmp/.X11-unix:/tmp/.X11-unix \
-v $PWD:/host/ \
--device=/dev/video0:/dev/video0 \
alexwitt23/mdp_tracker:latest /bin/bash
