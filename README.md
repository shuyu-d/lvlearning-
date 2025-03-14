# LVlearn: learning measurement models via subspace identification and clustering 

This repository contains an implementation of the LVlearn method for learning measurement models. 

## Requirements

- Python 3.6+
- numpy
- scipy
- python-igraph: Install [igraph C core](https://igraph.org/c/) and `pkg-config` first.
- `NOTEARS/utils.py` - graph simulation, data simulation, and accuracy evaluation from [Zheng et al. 2018]

Contact: shuyu.dong-at-centralesupelec.fr  

## Running a demo

```bash
gt=ER
deg=2
d=100
k=10

wmode=1        # Measurement model strengths are real-valued (positive and negative weights)
noise_var=2.0  # Noise variance 
eps=0.07       # Parameter of LVlearn 

FDIR=outputs/bm_"$str"_"$gt""$deg"_d"$d" 
mkdir -p $FDIR

$ python run_lvlearn.py runwho=syn-ev algo=lvlearn-EVmodelSel opts=5,"$wmode","$noise_var","$eps" ds=$d k=$k graph_type=$gt sem_type=gauss degs=$deg rnd=$rnd SEED=$j fdir=$FDIR fout=res_rnd"$rnd"_seed"$j" verbo=1 > $FDIR/screenlog1_rnd"$rnd"_seed"$j"_eps"$eps".txt
```




