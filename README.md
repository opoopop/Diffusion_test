# Diffusion_test

**Command_helper.md**  has the commands for the experiment.

**fusion_style_sample.py** makes all the samples mixed with another class. Put this file under **scripts/** .Change the class by changing fix_number into other numbers.

```python
    def cond_fn(x, t, y=None):
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            fix_number=1
            #fixed_y = th.ones((x.shape[0],), dtype=th.long, device=x.device)  
            fixed_y = th.full((x.shape[0],), fix_number, dtype=th.long, device=x.device)
            logits = classifier(x_in, t)  # logits
            probabilities = F.softmax(logits, dim=-1)  
            log_probs = F.log_softmax(logits, dim=-1)  #  log-softmax

            selected = log_probs[range(len(logits)), fixed_y.view(-1)]  
            prob_class_1 = probabilities[:, fix_number].item()  

            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

```

**data_prepare.ipynb** has the notebook to load the image from npz file and calculate the fid or IS score. 
