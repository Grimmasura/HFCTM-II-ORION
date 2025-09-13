# Arch packaging for HFCTM-II-ORION

## Quick start (AUR-style build)
```bash
sudo pacman -S --needed base-devel git
git clone https://github.com/Grimmasura/HFCTM-II-ORION.git
cd HFCTM-II-ORION
makepkg -si
```

## Run
```bash
sudo systemctl enable --now orion-api.service
# or run manually
ORION_PORT=8080 orion-api
```

## Where things go
- Code: `/opt/hfctm-ii-orion`
- Launcher: `/usr/bin/orion-api`
- Service: `/usr/lib/systemd/system/orion-api.service`
- Optional env: `/etc/orion/orion.env`

## Dependencies
- Required: `python`, `python-fastapi`, `python-uvicorn`, `python-numpy`, `python-pydantic`, `python-httpx`, `python-psutil`
- Optional: `python-scipy`, `python-scikit-learn`, `python-torch`, `python-prometheus-client`
