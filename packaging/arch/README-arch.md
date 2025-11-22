# Arch packaging for HFCTM-II-ORION

## Quick start
sudo pacman -S --needed base-devel git
git clone https://github.com/Grimmasura/HFCTM-II-ORION.git
cd HFCTM-II-ORION
makepkg -si

## Run
sudo systemctl enable --now orion-api.service
# or run directly:
ORION_PORT=8080 orion-api

## Paths
- Code: /opt/hfctm-ii-orion
- CLI: /usr/bin/orion-api
- Service: /usr/lib/systemd/system/orion-api.service
- Env: /etc/orion/orion.env

## Optional deps
sudo pacman -S python-pytorch  # or python-pytorch-cuda
