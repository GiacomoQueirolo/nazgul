WEBSITE="https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots//download?"
# define user and password here
USR=$1
PW=$2
# define the simulation to consider
SIM=$3 #"RefL050N0752"

#define where to store the data
DATADIR=$4
#define where to link the data
SUBDATADIR=$5
CRTDIR=`pwd`
mkdir -p $DATADIR
cd $DATADIR
# the snapshots are for now limited between 11 to 28
# so that the redshift for now is only 3.53<=z<=0
# see https://dataweb.cosma.dur.ac.uk:8443/eagle-snapshots/
# (require usr and pwd as defined up)
MIN_SNAP=$6
MAX_SNAP=$7

for i in `seq $MIN_SNAP $MAX_SNAP`
do 
    snap=$(printf %03d $i)
    wget --user=$USR --password=$PW --content-disposition "${WEBSITE}run=${SIM}&snapnum=$i" 
    tar -xvf ${SIM}_snap_${snap}.tar
done
# check that all directories have 16 files
cd $SIM

for i in `ls -d */` 
do  
        NF=`ls $i/* | wc -l` 
        if (($NF!=16)) 
                then echo "${i} doesn't have 16 files, check it" 
        fi  
done

# we can then remove the *tar files
cd -
rm -rf *.tar

# move back to the nazgul main dir
cd $CRTDIR

# link the snapshots in the correct directories
mkdir -p ${SUBDATADIR}
cd ${SUBDATADIR}
mkdir -p EAGLE
cd EAGLE
mkdir -p ${SIM}
cd ${SIM}
for i in `seq $MIN_SNAP $MAX_SNAP`
do 
    snap=$(printf %03d $i)
    snap_dir=snap_$snap
    mkdir -p $snap_dir
    cd $snap_dir
    mkdir -p ParticleData
    cd ParticleData
    ln -s ${DATADIR}/${SIM}/snapshot_${snap}_z???p???/snap_${snap}_z*.hdf5 .
    cd ../../
done
