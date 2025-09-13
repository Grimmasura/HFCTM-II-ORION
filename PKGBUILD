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

# Include auxiliary files so they exist in ${srcdir} at package() time
source=(
  "git+${url}.git"
  "orion-api"
  "orion-api.service"
  "README-arch.md"
  "orion.env.example"
)
sha256sums=('SKIP' 'SKIP' 'SKIP' 'SKIP' 'SKIP')

pkgver() {
  cd "${srcdir}/HFCTM-II-ORION"
  echo "r$(git rev-list --count HEAD).$(git rev-parse --short HEAD)"
}

prepare() {
  chmod +x "${srcdir}/orion-api"
}

build() {
  : # nothing to build; python project without pyproject
}

package() {
  # Code payload
  install -d "${pkgdir}/opt/hfctm-ii-orion"
  cp -a "${srcdir}/HFCTM-II-ORION"/. "${pkgdir}/opt/hfctm-ii-orion/"

  # CLI wrapper
  install -Dm755 "${srcdir}/orion-api" \
    "${pkgdir}/usr/bin/orion-api"

  # systemd unit
  install -Dm644 "${srcdir}/orion-api.service" \
    "${pkgdir}/usr/lib/systemd/system/orion-api.service"

  # env example
  install -Dm644 "${srcdir}/orion.env.example" \
    "${pkgdir}/etc/orion/orion.env"

  # docs
  install -Dm644 "${srcdir}/README-arch.md" \
    "${pkgdir}/usr/share/doc/${pkgname}/README-arch.md"
}
