# Checkpoints

Model weights are intentionally not tracked in Git.

Download the pretrained CASIAL checkpoint and place it here before testing:

https://drive.google.com/file/d/1fOi15BFPv1kKj7q8WlJetZwttMcfFrvu/view?usp=share_link

```bash
mkdir -p checkpoints
cp /path/to/300.pt checkpoints/300.pt
```

The default test command expects:

```text
checkpoints/300.pt
```
