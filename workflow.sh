#!/bin/bash
export IMAGE_PLATYPUS=maju116/platypus
export TAG_PLATYPUS=0.1.1
REPO_HOST_DIR=$(pwd)

remove_and_stop_rstudio () {
  container_id=$(docker ps -q -a -f name=platypus)
  container_id_length=${#container_id}
  if [[ $container_id_length > 0 ]]; then
    docker stop platypus
    docker rm platypus
  fi
}

echo "Using image $IMAGE_PLATYPUS:$TAG_PLATYPUS"

if [ "$1" == "build" ]; then

  cd environment
  docker build -f Dockerfile -t $IMAGE_PLATYPUS:$TAG_PLATYPUS .
  cd ..

elif [ "$1" == "pull" ]; then

  docker pull $IMAGE_PLATYPUS:$TAG_PLATYPUS

elif [ "$1" == "push" ]; then

  docker push $IMAGE_PLATYPUS:$TAG_PLATYPUS

elif [ "$1" == "platypus" ]; then

  remove_and_stop_rstudio

  docker run -d --name platypus -e USERID=$UID -e ROOT=true -p 8787:8787 -v "$REPO_HOST_DIR:/mnt" $IMAGE_PLATYPUS:$TAG_PLATYPUS

  echo "Rstudio running on: http://localhost:8787"
  xdg-open http://localhost:8787 > /dev/null

elif [ "$1" == "stop" ]; then

  remove_and_stop_rstudio
  echo "Container stopped and removed"

else
  echo "Usage: ./workflow.sh [param]"
  echo
  echo "Params:"
  echo
  echo "   build - quick build of new image from Dockerfile based on base image"
  echo "   pull - get image from Docker Hub"
  echo "   push - push image to Docker Hub"
  echo "   platypus - run image as daemon and start Rstudio in a browser"
  echo "   stop - stop running container"
  echo
fi
