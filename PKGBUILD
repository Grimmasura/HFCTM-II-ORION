# Maintainer: Joshua-friendly recipe via ChatGPT
pkgname=hfctm-ii-orion-git
pkgver=0
pkgrel=2
pkgdesc="Omniversal Recursive Intelligence for Ontological Navigation (HFCTM-II-ORION)"
arch=('x86_64')
url='https://github.com/Grimmasura/HFCTM-II-ORION'
license=('GPL2')

depends=(
  'python'
  'python-fastapi'
  'uvicorn'           # Arch package name (NOT python-uvicorn)
  'python-numpy'
  'python-pydantic'
  'python-httpx'
  'python-psutil'
)
optdepends=(
  'python-scipy: quantum stabilizer & ODE solvers'
  'python-scikit-learn: metrics and validation utilities'
  'python-torch: deep learning backends'
  'python-prometheus-client: /metrics endpoint'
)
makedepends=('git')

source=("git+${url}.git")
sha256sums=('SKIP')

pkgver() {
  cd "${srcdir}/HFCTM-II-ORION"
  echo "r$(git rev-list --count HEAD).$(git rev-parse --short HEAD)"
}

prepare() {
  cd "${srcdir}/HFCTM-II-ORION"
  chmod +x packaging/arch/orion-api
}

build() {
  : # nothing to build; python project without pyproject
}

package() {
  cd "${srcdir}/HFCTM-II-ORION"

  # Code payload
  install -d "${pkgdir}/opt/hfctm-ii-orion"
  cp -a . "${pkgdir}/opt/hfctm-ii-orion/"

  # CLI wrapper
  install -Dm755 "packaging/arch/orion-api" \
    "${pkgdir}/usr/bin/orion-api"

  # systemd unit
  install -Dm644 "packaging/arch/orion-api.service" \
    "${pkgdir}/usr/lib/systemd/system/orion-api.service"

  # env example
  install -Dm644 "packaging/arch/orion.env.example" \
    "${pkgdir}/etc/orion/orion.env"

  # docs
  install -Dm644 "packaging/arch/README-arch.md" \
    "${pkgdir}/usr/share/doc/${pkgname}/README-arch.md"
}
