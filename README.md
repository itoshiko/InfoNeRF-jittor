# InfoNeRF-jittor
Jittor implementation of [InfoNeRF: Ray Entropy Minimization for Few-Shot Neural Volume Rendering](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_InfoNeRF_Ray_Entropy_Minimization_for_Few-Shot_Neural_Volume_Rendering_CVPR_2022_paper.html)
[also a course project of "Computer Graphics (计算机图形学)" of CS, Tsinghua University]

## Environment
Following packages are required: jittor, tqdm, opencv-python, numpy, toml.

*For Jittor installation, please check [Jittor](https://github.com/Jittor/jittor.git)*

## Training and Testing
Download the data [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1). Place the downloaded dataset according to the following directory structure:
```
├── configs  
│   ├── ...    
│                                                                                      
├── data 
|   ├── nerf_synthetic
|   |   └── lego
|   |   └── ship    # downloaded synthetic dataset
|   |   └── ...
```

To train a InfoNeRF on the example lego scene:
```bash
python train_nerf.py
```

To train and test on other scenes, please write your own config files and change the path of config files at the entry of the program:
```python
# change to your own config files
train_cfg = load_config('configs/lego.toml', 'configs/base.toml')
trainer = Trainer(train_cfg)
trainer.train()
trainer.run_testset(save_path, 1)
```


## Reference
* [Pytorch implementation of NeRF](https://github.com/yenchenlin/nerf-pytorch.git)
* [Pytorch implementation of InfoNeRF](https://github.com/mjmjeong/InfoNeRF.git)

