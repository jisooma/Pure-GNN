# Pure-GNN: A Lightweight Purified Graph Neural Network against Adversarial Attacks

 ![image-20240319202727032](https://raw.githubusercontent.com/mazhixiu09/pictures/master/blogimg/202403192033655.png)


## Requirements

```bash
python=3.8.15
pytorch=1.12.0
deeprobust==0.2.6
numpy=1.23.5
tqdm=4.65.0
```

## 1、Attacks(generate poisoned datasets)

```bash
python ./generate_attack/generate_global attack.py
python ./generate_attack/generate_targeted_attack.py
```

## 2、Other's Defense

```bash
python ./compare_denfense/defense.py
```

## 3、Pure-GNN

```bash
python ./model/Pure_GNN_test.py
```




