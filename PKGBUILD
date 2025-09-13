# Maintainer: Joshua-friendly recipe via ChatGPT
pkgname=hfctm-ii-orion-git
pkgver=0
pkgrel=1
pkgdesc="Omniversal Recursive Intelligence for Ontological Navigation (HFCTM-II-ORION)"
arch=('x86_64')
url='https://github.com/Grimmasura/HFCTM-II-ORION'
license=('GPL2')
depends=(
  'python'
  'python-fastapi'
  'python-uvicorn'
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
md5sums=('SKIP')

pkgver() {
  cd "${srcdir}/HFCTM-II-ORION"
  echo "r$(git rev-list --count HEAD).$(git rev-parse --short HEAD)"
}

build() {
  cd "${srcdir}/HFCTM-II-ORION"
  # nothing to build; python project without pyproject
  :
}

package() {
  install -d "${pkgdir}/opt/hfctm-ii-orion"
  cp -a "${srcdir}/HFCTM-II-ORION"/. "${pkgdir}/opt/hfctm-ii-orion/"

  # wrapper
  install -d "${pkgdir}/usr/bin"
  install -m755 "${srcdir}/orion-api" "${pkgdir}/usr/bin/orion-api"

  # systemd unit
  install -d "${pkgdir}/usr/lib/systemd/system"
  install -m644 "${srcdir}/orion-api.service" "${pkgdir}/usr/lib/systemd/system/orion-api.service"

  # optional env dir
  install -d "${pkgdir}/etc/orion"
  install -m644 "${srcdir}/orion.env.example" "${pkgdir}/etc/orion/orion.env"

  # docs
  install -d "${pkgdir}/usr/share/doc/${pkgname}"
  install -m644 "${srcdir}/README-arch.md" "${pkgdir}/usr/share/doc/${pkgname}/README-arch.md"
}
