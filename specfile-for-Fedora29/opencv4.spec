#global indice   a
%undefine _strict_symbol_defs_build
%bcond_with     tests
%bcond_with     ffmpeg
%bcond_without  gstreamer
%bcond_with     eigen2
%bcond_without  eigen3
%ifnarch ppc64le
%bcond_without  opencl
%else
# https://bugzilla.redhat.com/show_bug.cgi?id=1487174
%bcond_with     opencl
%endif
%ifarch %{ix86} x86_64 %{arm}
%bcond_without  openni
%else
# we dont have openni in other archs
%bcond_with     openni
%endif
%bcond_without  tbb
%bcond_with     cuda
%bcond_with     xine
# Atlas need (missing: Atlas_CLAPACK_INCLUDE_DIR Atlas_CBLAS_LIBRARY Atlas_BLAS_LIBRARY Atlas_LAPACK_LIBRARY)
# LAPACK may use atlas or openblas since now it detect openblas, atlas is not used anyway, more info please
# check OpenCVFindLAPACK.cmake
%bcond_with     atlas
%bcond_without  openblas
%bcond_without  gdcm
#VTK support disabled. Incompatible combination: OpenCV + Qt5 and VTK ver.7.1.1 + Qt4
%bcond_with     vtk
%ifarch %{ix86} x86_64
%bcond_without  libmfx
%else
%bcond_with     libmfx
%endif
%bcond_without  clp
%bcond_without  va

%global srcname opencv
%global abiver  4.1

# Required because opencv-core has lot of spurious dependencies
# (despite supposed to be "-core")
# TODO: to be fixed properly upstream
# https://github.com/opencv/opencv/issues/7001
%global optflags %(echo %{optflags} -Wl,--as-needed )

Name:           opencv
Version:        4.1.0
Release:        1%{?dist}
Summary:        Collection of algorithms for computer vision
# This is normal three clause BSD.
License:        BSD
URL:            http://opencv.org
# RUN opencv-clean.sh TO PREPARE TARBALLS FOR FEDORA
#
# Need to remove copyrighted lena.jpg images from tarball (rhbz#1295173)
# and SIFT/SURF from tarball, due to legal concerns.
#

BuildRequires:  libtool
BuildRequires:  cmake >= 2.6.3
BuildRequires:  chrpath
%{?with_eigen2:BuildRequires:  eigen2-devel}
%{?with_eigen3:BuildRequires:  eigen3-devel}
BuildRequires:  gtk3-devel
BuildRequires:  libtheora-devel
BuildRequires:  libvorbis-devel
%if 0%{?fedora} || 0%{?rhel} > 7
%ifnarch s390 s390x
BuildRequires:  libraw1394-devel
BuildRequires:  libdc1394-devel
%endif
%endif
BuildRequires:  jasper-devel
BuildRequires:  libjpeg-devel
BuildRequires:  libpng-devel
BuildRequires:  libtiff-devel
BuildRequires:  libGL-devel
BuildRequires:  libv4l-devel
BuildRequires:  gtkglext-devel
BuildRequires:  OpenEXR-devel
%{?with_openni:
BuildRequires:  openni-devel
BuildRequires:  openni-primesense
}
%{?with_tbb:
BuildRequires:  tbb-devel
}
BuildRequires:  zlib-devel pkgconfig
BuildRequires:  python2-devel
BuildRequires:  python3-devel
BuildRequires:  pylint
BuildRequires:  python2-numpy
BuildRequires:  python3-numpy
BuildRequires:  swig >= 1.3.24
%{?with_ffmpeg:BuildRequires:  ffmpeg-devel >= 0.4.9}
%if 0%{?fedora} || 0%{?rhel} > 7
%{?with_gstreamer:BuildRequires:  gstreamer1-devel gstreamer1-plugins-base-devel}
%else
%{?with_gstreamer:BuildRequires:  gstreamer-devel gstreamer-plugins-base-devel}
%endif
%{?with_xine:BuildRequires:  xine-lib-devel}
%{?with_opencl:BuildRequires:  opencl-headers}
BuildRequires:  libgphoto2-devel
BuildRequires:  libwebp-devel
BuildRequires:  tesseract-devel
BuildRequires:  protobuf-devel
BuildRequires:  gdal-devel
BuildRequires:  glog-devel
BuildRequires:  doxygen
BuildRequires:  python2-beautifulsoup4
#for doc/doxygen/bib2xhtml.pl
BuildRequires:  perl-open
BuildRequires:  gflags-devel
BuildRequires:  SFML-devel
BuildRequires:  libucil-devel
BuildRequires:  qt5-qtbase-devel
BuildRequires:  mesa-libGL-devel
BuildRequires:  mesa-libGLU-devel
BuildRequires:  hdf5-devel
%{?with_vtk:BuildRequires: vtk-devel}
%{?with_atlas:BuildRequires: atlas-devel}
#ceres-solver-devel push eigen3-devel and tbb-devel
%{?with_tbb:
  %{?with_eigen3:
BuildRequires:  ceres-solver-devel
  }
}
%{?with_openblas:
BuildRequires:  openblas-devel
BuildRequires:  blas-devel
BuildRequires:  lapack-devel
}
%{?with_gdcm:BuildRequires: gdcm-devel}
%{?with_libmfx:BuildRequires:  libmfx-devel}
%{?with_clp:BuildRequires:  coin-or-Clp-devel}
%{?with_va:BuildRequires:   libva-devel}

Requires:       opencv-core%{_isa} = %{version}-%{release}
BuildRequires:       gcc, gcc-c++

%description
OpenCV means Intel® Open Source Computer Vision Library. It is a collection of
C functions and a few C++ classes that implement some popular Image Processing
and Computer Vision algorithms.


%package        core
Summary:        OpenCV core libraries

%description    core
This package contains the OpenCV C/C++ core libraries.


%package        devel
Summary:        Development files for using the OpenCV library
Requires:       %{name}%{_isa} = %{version}-%{release}
Requires:       %{name}-contrib%{_isa} = %{version}-%{release}

%description    devel
This package contains the OpenCV C/C++ library and header files, as well as
documentation. It should be installed if you want to develop programs that
will use the OpenCV library. You should consider installing opencv-doc
package.


%package        doc
Summary:        docs files
Requires:       opencv-devel = %{version}-%{release}
BuildArch:      noarch
Provides:       %{name}-devel-docs = %{version}-%{release}
Obsoletes:      %{name}-devel-docs < %{version}-%{release}

%description    doc
This package contains the OpenCV documentation, samples and examples programs.


%package        -n python2-opencv
Summary:        Python2 bindings for apps which use OpenCV
Requires:       opencv%{_isa} = %{version}-%{release}
Requires:       python2-numpy
%{?python_provide:%python_provide python2-%{srcname}}
# Remove before F30
Provides:       %{name}-python = %{version}-%{release}
Provides:       %{name}-python%{?_isa} = %{version}-%{release}
Obsoletes:      %{name}-python < %{version}-%{release}

%description    -n python2-opencv
This package contains Python bindings for the OpenCV library.


%package        -n python3-opencv
Summary:        Python3 bindings for apps which use OpenCV
Requires:       opencv%{_isa} = %{version}-%{release}
Requires:       python3-numpy
%{?python_provide:%python_provide python3-%{srcname}}
# Remove before F30
Provides:       %{name}-python3 = %{version}-%{release}
Provides:       %{name}-python3%{?_isa} = %{version}-%{release}
Obsoletes:      %{name}-python3 < %{version}-%{release}

%description    -n python3-opencv
This package contains Python3 bindings for the OpenCV library.


%package        contrib
Summary:        OpenCV contributed functionality

%description    contrib
This package is intended for development of so-called "extra" modules, contributed
functionality. New modules quite often do not have stable API, and they are not
well-tested. Thus, they shouldn't be released as a part of official OpenCV
distribution, since the library maintains binary compatibility, and tries
to provide decent performance and stability.

%prep
%setup -D -c -T -n opencv
if [ ! -d opencv ]; then
    git clone https://github.com/opencv/opencv.git
fi
if [ ! -d opencv_contrib ]; then
    git clone https://github.com/opencv/opencv_contrib.git
fi

cd opencv
git checkout master
git pull

cd ../opencv_contrib
git checkout master
git pull



%build

exit

cd opencv
# enabled by default if libraries are presents at build time:
# GTK, GSTREAMER, 1394, V4L, eigen3
# non available on Fedora: FFMPEG, XINE
mkdir -p build
pushd build

# disabling IPP because it is closed source library from intel

%cmake CMAKE_VERBOSE=1 \
 -DWITH_IPP=OFF \
 -DWITH_ITT=OFF \
 -DWITH_QT=ON \
 -DWITH_OPENGL=ON \
 -DWITH_GDAL=ON \
 -DWITH_UNICAP=ON \
 -DCMAKE_SKIP_RPATH=ON \
 -DWITH_CAROTENE=OFF \
 -DENABLE_PRECOMPILED_HEADERS=OFF \
 -DCMAKE_BUILD_TYPE=ReleaseWithDebInfo \
 -DBUILD_opencv_java=OFF \
 %{?with_tbb: -DWITH_TBB=ON } \
 %{!?with_gstreamer: -DWITH_GSTREAMER=OFF } \
 %{!?with_ffmpeg: -DWITH_FFMPEG=OFF } \
 %{?with_cuda: \
 -DWITH_CUDA=ON \
 -DCUDA_TOOLKIT_ROOT_DIR=%{?_cuda_topdir} \
 -DCUDA_VERBOSE_BUILD=ON \
 -DCUDA_PROPAGATE_HOST_FLAGS=OFF \
 } \
 %{?with_openni: -DWITH_OPENNI=ON } \
 %{!?with_xine: -DWITH_XINE=OFF } \
 -DBUILD_DOCS=ON \
 -DBUILD_EXAMPLES=ON \
 -DINSTALL_C_EXAMPLES=ON \
 -DINSTALL_PYTHON_EXAMPLES=ON \
 -DENABLE_PYLINT=ON \
 -DBUILD_PROTOBUF=OFF \
 -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
 -DWITH_LIBV4L=ON \
 -DWITH_OPENMP=ON \
 -DENABLE_PKG_CONFIG=OFF \
 -DOPENCV_GENERATE_PKGCONFIG=YES \
 %{?with_gdcm: -DWITH_GDCM=ON } \
 %{?with_libmfx: -DWITH_MFX=ON } \
 %{?with_clp: -DWITH_CLP=ON } \
 %{?with_va: -DWITH_VA=ON } \
 ..

%make_build VERBOSE=1


%install
cd opencv
%make_install -C build
find %{buildroot} -name '*.la' -delete

%check
# Check fails since we don't support most video
# read/write capability and we don't provide a display
# ARGS=-V increases output verbosity
# Make test is unavailble as of 2.3.1
#ifnarch ppc64
%if %{with tests}
pushd build
    LD_LIBRARY_PATH=%{_builddir}/%{name}-%{version}/build/lib:$LD_LIBARY_PATH make test ARGS=-V || :
popd
%endif
#endif


%ldconfig_scriptlets core

%ldconfig_scriptlets contrib


%files
#doc README.md
#license LICENSE
%license %{_datadir}/licenses/opencv4/*
%{_bindir}/opencv_*
%{_bindir}/setup*
%dir %{_datadir}/opencv4
%{_datadir}/opencv4/haarcascades
%{_datadir}/opencv4/lbpcascades
%{_datadir}/opencv4/valgrind*

%files core
%{_libdir}/libopencv_*
%{_libdir}/opencv4

%files devel
%{_includedir}/opencv4
%{_libdir}/lib*.so
%{_libdir}/pkgconfig/opencv4.pc
%dir %{_libdir}/cmake/opencv4
%{_libdir}/cmake/opencv4/*.cmake

%files doc
%{_datadir}/opencv4/samples
#{_datadir}/opencv4/doc

%files -n python2-opencv
%{python2_sitearch}/cv*

%files -n python3-opencv
%{python3_sitearch}/cv*

%files contrib
%{_libdir}/libopencv_*

%changelog
* Wed Apr 17 2019 Mark Fassler <mfassler@users.noreply.github.com> - 4.1.0-1
- Build the latest 4.x version, with DNN enabled

* Fri Jul 13 2018 Fedora Release Engineering <releng@fedoraproject.org> - 3.4.1-6
- Rebuilt for https://fedoraproject.org/wiki/Fedora_29_Mass_Rebuild


