# Experiments comparing the performance of GNN and MLP models of cancer prognosis

## About
The scripts in this repository were used in experiments that compared the quality of predictions made by GNN- and 
MLP-based prognostic models of cancer. These experiments extended the work of Liang _et al._ in their 2022 paper _Risk 
stratification and pathway analysis based on graph neural network and interpretable algorithm_. Whereas Liang _et al_.
modeled the probability that a patient would survive at least 3 years beyond their diagnosis date, the experiments 
herein modelled not only single-time prognosis, but also survival probability distributions for all times 
post-diagnosis. These experiments also provide a more rigorous assessment of the effect of feature selection on the 
models' predictions, and model prognosis in several primary tumor types that were not studied by Liang _et al_.

## Cloning and setting up the repository locally
Some of the files needed to run the experiments are not included in this repository and can be obtained elsewhere. 
After cloning the `gnnsurvival` repository, get the remaining files from `BioAI-kits/PathGNN.git`.
```shell
PROJECT_NAME=gnnsurvival
PROJECT_DIR=/Users/${USER}/Projects/${PROJECT_NAME}
git clone git@github.com:BioAI-kits/PathGNN.git
cd PathGNN/Pathway
unzip pathways.zip
cp -r PathGNN/Pathway ${PROJECT_DIR}/data
```

## Running as a container on a remote host (example)
There are two Dockerfiles in the `container` branch. The first Dockerfile, `Dockerfile.setup`, scrapes TCGA gene 
expression data and performs basic preprocessing on data that are shared by all experiments. The second Dockerfile, 
`Dockerfile.main`, runs an experiment. Multiple different experiments can be run concurrently without conflict.

Assuming you've cloned the repository to the local host, start by copying required files to the remote host, 
building the preprocessing container image, and transferring it to the remote host.
```shell
PROJECT_NAME=gnnsurvival
PROJECT_DIR=/Users/${USER}/Projects/${PROJECT_NAME}
REMOTE_USER=remote_user
REMOTE_HOST=remote_host
REMOTE_HOST_PLATFORM=linux/amd64
REMOTE_DEST=/home/${REMOTE_USER}

cd ${PROJECT_DIR}

scp -r data/Pathway ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/${PROJECT_NAME}/data/
scp data/tcga_clin.csv ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/${PROJECT_NAME}/data/
scp data/feature_map.csv ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/${PROJECT_NAME}/data/

TAG=data-setup
TARBALL=${PROJECT_DIR}/${PROJECT_NAME}_${TAG}.tar
git checkout container
docker build -t ${PROJECT_NAME}:${TAG} -f Dockerfile.setup --platform=${REMOTE_HOST_PLATFORM} .
docker save -o ${TARBALL} ${PROJECT_NAME}:${TAG}
scp ${TARBALL} ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/
docker rmi ${PROJECT_NAME}:${TAG}
```
Create a new local Git branch that contains scripts for an experiment you want to run and `checkout` this newly 
created branch. Next, build the experiment's container image and transfer it to the remote host.
```shell
EXPT_NAME=experiment1
TARBALL=${PROJECT_DIR}/${PROJECT_NAME}_${EXPT_NAME}.tar
git checkout ${EXPT_NAME}
docker build -t ${PROJECT_NAME}:${EXPT_NAME} -f Dockerfile.main --platform=${REMOTE_HOST_PLATFORM} .
docker save -o ${TARBALL} ${PROJECT_NAME}:${EXPT_NAME}
scp ${TARBALL} ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DEST}/
docker rmi ${PROJECT_NAME}:${EXPT_NAME}
```
Connect to the host via SSH. From the remote terminal session, set up the project directory.
```shell
PROJECT_NAME=gnnsurvival
PROJECT_DIR=/home/${USER}/${PROJECT_NAME}

[ ! -e ${PROJECT_DIR} ] && mkdir ${PROJECT_DIR}
[ ! -e ${PROJECT_DIR}/data ] && mkdir ${PROJECT_DIR}/data
```
Run the data setup container on the remote host. This will create all the shared data needed to run the experiment.
The data will be written to a shared volume that will be accessed during the experiment.
```shell
TAG=data-setup
TARBALL=/home/${USER}/${PROJECT_NAME}_${TAG}.tar
docker load < ${TARBALL}
docker run \
  -v ${PROJECT_DIR}/data:/${PROJECT_NAME}/data \
  -v ${PROJECT_DIR}/${TAG} \
  ${PROJECT_NAME}:${TAG}
```
Unpack the experiment container image from the tarball.
```shell
EXPT_NAME=experiment1
TARBALL=/home/${USER}/${PROJECT_NAME}_${EXPT_NAME}.tar
docker load < ${TARBALL}
```
Now, with the images unpacked and ready to go on the remote host, run a containerized experiment for a particular 
cancer type. Note that sufficient shared memory must be allocated to run the container, at least 64GB, possibly more.
256GB is used in the example below.

The experiment can be run in parallel with different cancer types if the remote host is powerful enough to 
handle it. Open up one SSH terminal per cancer type and create an environment variable that stores the cancer type 
and run the experiment. Note that different experiments can be run serially with the same cancer type in this SSH 
terminal session, but in this example only one experiment is executed.
```shell
CANCER=LUAD
PROJECT_NAME=gnnsurvival
PROJECT_DIR=/home/${USER}/${PROJECT_NAME}
SHM_SIZE=256gb

EXPT_NAME=experiment1
[ ! -e ${PROJECT_DIR}/${EXPT_NAME} ] && mkdir ${PROJECT_DIR}/${EXPT_NAME}
docker run \
  --shm-size=${SHM_SIZE} \
  -v /dev/shm:/dev/shm \
  -v ${PROJECT_DIR}/data:/${PROJECT_NAME}/data \
  -v ${PROJECT_DIR}/${EXPT_NAME}:/${PROJECT_NAME}/${EXPT_NAME} \
  ${PROJECT_NAME}:${EXPT_NAME} -e ${EXPT_NAME} -t ${CANCER}
```

## Removing files that were created by Docker in a mounted volume: a note for users without root access
Files created by a Docker container in a mounted file system are owned by root. If you don't have root access, you 
can't remove files owned by root. A workaround is to start the container in the background and enter a bash session 
as root inside the container and remove the files within that bash session. N.B. This should be done before removing 
any other files from the mounted directory that aren't owned by root but are used by the container.

First, build a Docker container that will run forever. Working on the remote host via SSH, create a shell script that 
sleeps indefinitely:
```shell
mkdir /home/${USER}/sleep-forever && cd /home/${USER}/sleep-forever
tee -a sleep-forever.sh << END
#! /bin/sh
while true
do
  sleep 1
done
END
```
Create a Dockerfile with the following contents:
```dockerfile
# syntax=docker/dockerfile:1
FROM alpine:latest
WORKDIR /sleep-forever
RUN apk add --no-cache bash
COPY . .
RUN chmod u+x sleep-forever.sh
CMD ["bash", "sleep-forever.sh"]
```
Build the image and run the container in the background.
```shell
PROJECT_NAME=gnnsurvival
PROJECT_DIR=/home/${USER}/${PROJECT_NAME}

docker build -t $sleep-forever:latest -f Dockerfile .
docker run -d -i -t -v ${PROJECT_DIR}:/${PROJECT_NAME} sleep-forever:latest
CONTAINER_ID=$(docker ps -qf "ancestor=sleep-forever:latest")
docker exec -i -t --privileged --user root ${CONTAINER_ID} /bin/bash
```
You should now be root in a bash session in the `sleep-forever` container. Remove the experiment data from here.
```bash
EXPT_NAME=experiment1
rm -rf ${EXPT_NAME} data
exit
```
Finally, remove the running container:
```shell
docker container rm ${CONTAINER_ID}
```
