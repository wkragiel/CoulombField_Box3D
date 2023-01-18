"""Input and output of states to various data formats.

Binary data is stored in one of the following formats denoted by their
extension:

wdata:
    Filename ends in ``.wdat``.  This is a simple binary format
    contiguous in C ordering.  In this format there is no information
    about the size, datatype, or shape of the data.  Thus, a
    corresponding metadata file is needed.
npy:
    Filename ends in ``.wdat``  This is the NPY format.  This includes
    the size and datatype, so one can infer the type of data without a
    metadata file.  We assume that the shape is one of:

    * ``(Nt,) + Nxyz``: Scalar data with `Nt` blocks each of shape ``Nxyz``.
    * ``(Nt, Nv) + Nxyz``: Vector data with ``Nt`` blocks, and ``Nv <= 3``
        components each.  Note: we assume ``Nx > 3`` so that these two cases
        can be distinguished by looking at ``shape[1]``.

Note: Both of these formats are simple binary representations, and hence can be loaded
as memory mapped files.  We do this by default in the current code so users can extract
slices as needed without loading the entire dataset.  Be aware that the data can be big,
so if you do anything that makes a copy of the array (like computing `np.sin(...)`),
this will load the entire array into memory.

See Also
--------
* https://numpy.org/doc/stable/reference/generated/numpy.memmap.html

"""
from collections import OrderedDict
import glob
import os.path
from warnings import warn

from datetime import datetime
import numpy as np

from pytz import timezone
import tzlocal

from zope.interface import implementer, Attribute, Interface


class IVar(Interface):
    """Interface for a variable descriptor.

    Instances may actually store the data, or may represent data on
    disk that has not yet been loaded.

    If data is provided, then it must have the shape:

    * Abscissa data: ``(N,)``
    * Scalar data: ``(Nt,) + Nxyz``
    * Vector data: ``(Nt, Nv) + Nxyz``

    Assumes that ``Nx > 3`` so that if ``shape[1] <= 3``, the data is vector with ``Nv``
    components.  Physically, one can have ``Nv == dim`` or ``Nv == 3``.  For example, a
    ``dim = 2`` dataset might still have ``Nv == 3`` vectors where the third component
    indicates that there is a persistent current in the z-direction.  Although the code
    allows ``Nv < dim``, this does not make much sense.
    """

    # Required attributes
    name = Attribute("Name of variable")
    data = Attribute("Array-like object with access to the data.")
    description = Attribute("Single-line description of the data.")
    unit = Attribute("String representation of units for the data.")
    filename = Attribute("Full path of the data file.")

    # Derived attributes: can be computed from data if present.
    descr = Attribute("NumPy data descriptor dtype.descr")
    shape = Attribute("Shape of the data so that wdat files can be read.")

    def __init__(
        name=None,
        data=None,
        description=None,
        unit="none",
        filename=None,
        descr=None,
        shape=None,
        **kwargs,
    ):
        """Initialize the object.

        Arguments
        ---------
        name : str
            Name of the variable.  If not provided, then the data
            should be set using kwarg syntax ``__init__(name=data)``.
        kwargs : {name: data}
            If no name is provided, then kwargs should have a single
            entry which is the data and the key is the name.
        filename : str
            Name of file (including path).

        description : str
            Single-line comment for variable.  Defaults to the name.
        unit : str
            Unit of data for metadata file.
        shape : tuple, None
            If no data is provided, then this is needed so that wdat
            files can be loaded.
        """

    def write_data(filename=None, force=False):
        """Write the data to disk.

        Arguments
        ---------
        filename : str, None
            Filename.  Data type is determined by the extension.
            Currently supported values are ``'wdat'`` (pure binary) and ``'npy'`` for the
            numpy NPY format.  Uses ``self.filename`` if not provided.
        force : bool
            If True, overwrite existing files, otherwise raise IOError
            if file exists.
        """


class IWData(Interface):
    """Interface of a complete dataset.

    Also used to define abscissa as follows::

        x = np.arange(Nx)*dx + x0
        t = np.arange(Nt)*dt + t0

    If not provided, the default values are ``x0 = -dx * (Nx // 2)`` so
    that the spatial abscissa are centered and include the origin and ``t0 = 0``. (This
    is the default behavior of :class:`mmfutils.math.bases.basis.PeriodicBasis` as over
    version 0.6.0.)

    >>> data = WData(Nxyz=(5,), Nt=4)
    >>> data.xyz
    [array([-2.5, -1.5, -0.5,  0.5,  1.5])]
    >>> data.t
    [0, 1, 2]
    """

    prefix = Attribute("Prefix for all data files")
    description = Attribute("Description of the data.")
    data_dir = Attribute("Location of data.")
    ext = Attribute("Default extension for data file format.")

    variables = Attribute("List of IVar instances for the variables")

    aliases = Attribute("Dictionary of aliases.")
    constants = Attribute("Dictionary of constants.")

    xyz = Attribute("Abscissa (x, y, z) broadcast appropriately.")
    dim = Attribute("Dimensions. 1: Nx, 2: Nx*Ny, 3: Nx*Ny*Nz")

    Nxyz = Attribute("Shape of data (Nx, Ny, Nz) and lattice shape.")
    xyz0 = Attribute("Offsets (x0, y0, z0).")

    dxyz = Attribute("Spacing (dx, dy, dz).  ``np.nan`` if not uniform.")

    t = Attribute("Times for each frame")

    t0 = Attribute("Time of initial frame.")
    dt = Attribute("Time steps between frames.  ``np.nan`` if not uniform.")

    infofile = Attribute("Name of infofile")

    def __init__(
        prefix="tmp",
        description="",
        data_dir=".",
        ext="wdat",
        dim=None,
        Nxyz=None,
        dxyz=(1, 1, 1),
        xyz0=None,
        xyz=None,
        Nt=0,
        t0=0,
        dt=1,
        t=None,
        variables=None,
        aliases=None,
        constants=None,
        check_data=True,
    ):
        """Constructor.

        Arguments
        ---------
        prefix : str
            Prefix for files.  Default is ``'tmp'`` allowing this class to
            be used to generate abscissa.
        description : str
            User-friendly description of the data.  Will be inserted as
            a comment in the metadata file.
        data_dir : str
            Path to the directory containing the data.
        ext : str
            Default extension for data file format used if variables do
            not specify something else.  Also used for abscissa.

        Nxyz : (int,)*dim

        Nxyz, dxyz, xyz0 : (int,)*dim, (float,)*dim, (float,)*dim or None
            If these are provided, then the abscissa ``xyz`` are computed
            with equal spacings ``x = np.arange(Nx)*dx + x0``.  Default
            offeset (if ``xyz0 == None``) is centered ``x0 = -dx * (Nx // 2)``.

            New in version 0.1.4, we allow `dim <= len(Nxyz)` etc.  In this case, `Nxyz` can
            still be fully specified, indicating that the underlying W-SLDA code used a
            certain number of plane-waves in the extra dimensions.
        xyz : (array_like,)*dim
            Alternatively, the abscissa can be provided and the
            previous properties will be computed (if defined).
        Nt, t0, dt : int, float, float
            If these are provided, then the times are equally spaced.
            ``t = np.arange(Nt)*dt + t0``.
        t : array_like
            Alternatively, the times can be provided and the previous
            properties will be computed.

        variables : [IVar]
            List of Variables.

        check_data : bool
            If True, then upon initialization try to load all the data and check
            that the dimensions are consistent.
            (New in version 0.1.4)
        """

    def get_metadata(header=None):
        """Return a string representation of the metadata file.

        Arguments
        ---------
        header : str
            Descriptive header to be added as a comment at the top of
            the metadata file before self.description.
        """

    def save(force=False):
        """Save data to disk.

        Arguments
        ---------
        force : bool
            If True, create needed directories and overwrite existing
            data files.  Otherwise, raise an IOError.
        """

    def load(infofile=None, full_prefix=None, check_data=True):
        """Load data from disk.

        Arguments
        ---------
        infofile : str
            If provided, then read this file and use the information
            contained within it to load the data.
        full_prefix : str
            Full prefix for data files, including directory if not in
            the current directory.  I.e. ``prefix='data_dir/run1'`` will
            look for data in the form of ``'data_dir/run1_*.*'`` and an
            infofile of the form ``'data_dir/run1.wtxt'``.

            The full_prefix will be split at the final path
            separation and what follows will be the ``prefix``.
        check_data : bool
            If `True` (default), then check that the data exists and is consistent with
            the description.  Can be set to `False` if data is missing.  One will then
            have an error later if the data is accessed.

        **No infofile option**

        Data can be provided without metadata if the following files
        are present::

            <full_prefix>_x.npy
            <full_prefix>_y.npy
            <full_prefix>_z.npy
            <full_prefix>_t.npy
            <full_prefix>_*.npy

        In his case, the abscissa will be determined by loading the
        first four files, and the remaining files will defined the
        variables.  No constants or links will be defined in this
        case.

        One other option is provided when ``full_prefix`` is a
        directory: i.e. ends with a path separator.  In this case, the
        directory will be assumed to contain a single set of data with
        a prefix determined by the filenames if the files above exist.
        """

    def __getattr__(key):
        """Convenience method for variable access.

        Returns the data of the named variable or the named
        constant, following aliases if defined."""

    def keys():
        """Return the valid keys."""


@implementer(IVar)
class Var(object):
    def __init__(
        self,
        name=None,
        data=None,
        description=None,
        filename=None,
        unit="none",
        descr=None,
        shape=None,
        **kwargs,
    ):
        if name is None:
            if data is not None:
                raise ValueError("Got data but no name.")
            if not len(kwargs) == 1:
                raise ValueError(
                    f"Must provide `name` or data as a kwarg: got {kwargs}"
                )
            name, data = kwargs.popitem()
        self.name = name
        self.description = description
        self.filename = filename
        self.unit = unit
        self._descr = descr
        self._data = data
        self.shape = shape
        self.init()

    def init(self):
        """Perform all initialization and checks.

        Called by the constructor, and before saving.
        """
        if self._data is not None:
            # Convert data and check some properties
            args = {}
            if self._descr is not None:
                args.update(dtype=np.dtype(self._descr))
            self._data = np.ascontiguousarray(self._data, **args)

        if self.description is None:
            self.description = self.name

    @property
    def data(self):
        """Return or load data."""
        if self._data is None:
            self._load_data()
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.init()

    @property
    def descr(self):
        if self._data is None:
            return self._descr
        else:
            descr = self._data.dtype.descr
            assert len(descr) == 1
            return descr[0][-1]

    @property
    def vector(self):
        return len(self.shape) > 1 and self.shape[1] <= 3

    @property
    def shape(self):
        if self._data is None:
            return self._shape
        else:
            return self._data.shape

    @shape.setter
    def shape(self, shape):
        if self._data is not None and shape is not None:
            if np.prod(shape) != np.prod(self._data.shape):
                raise ValueError(
                    f"Property shape={shape} incompatible "
                    + f"with data.shape={self._data.shape}"
                )
            self._data = self._data.reshape(shape)

        self._shape = shape

    def write_data(self, filename=None, force=False):
        """Write self.data to the specified file."""
        self.init()

        if filename is None:
            filename = self.filename

        if filename is None:
            raise ValueError("No filename specified in Var.")

        if self._data is None:
            raise ValueError(f"Missing data for '{self.name}'!")

        if os.path.exists(filename) and not force:
            raise IOError(f"File '{filename}' already exists!")

        A = self._data

        if filename.endswith(".wdat"):
            with open(filename, "wb") as fd:
                fd.write(A.tobytes())
        elif filename.endswith(".npy"):
            np.save(filename, A)
        else:
            raise NotImplementedError(f"Unsupported extension for '{filename}'")

    def _load_data(self):
        """Load the data from file."""
        if self.filename.endswith(".npy"):
            _data = np.load(self.filename, mmap_mode="r")
        elif self.filename.endswith(".wdat"):
            shape = self.shape
            _data = np.memmap(self.filename, dtype=np.dtype(self.descr), mode="r")
            if shape == (None,):  # _data is an abscissa
                shape = (len(_data),)

            if np.prod(_data.shape) == np.prod(shape):
                # Done, just reshape
                pass
            elif self.vector:
                Nt, Nv = self.shape[:2]
                dim = len(self.shape[2:])
                assert dim <= Nv
                NtNvNxyz = np.prod(_data.shape)
                Nxyz = np.prod(self.shape[2:])                
                NtNv = NtNvNxyz // Nxyz
                Nv = NtNv // Nt

                if (NtNvNxyz % Nxyz == 0) and (NtNv % Nt == 0) and dim <= Nv:
                    Nt = NtNv // Nv
                elif not (NtNvNxyz % Nxyz == 0):
                    # Inconsistent data size
                    raise ValueError(
                        f"Shape of data in '{self.filename}' inconsistent "
                        + f"with shape={shape}: NtNv={NtNvNxyz / Nxyz} must be an integer.")
                else:
                    # Possibly incomplete data
                    for Nv in range(dim, 4):
                        if (NtNv % Nv == 0) and NtNv // Nv <= Nt:
                            Nt = NtNv // Nv
                            break
                    if Nv == 4:
                        raise ValueError(
                            f"Shape of incomplete data in '{self.filename}' inconsistent "
                            + f"with shape={shape}: NvNt={NtNvNxyz / Nxyz}.  Must have "
                            + f"integer Nt>={Nt} and {dim}<=Nv<=3."
                    )
                self._shape = shape = (Nt, Nv) + shape[2:]
            else:
                # Try loading the data anyway, even if it is incomplete.
                NtNxyz = np.prod(_data.shape)
                Nxyz = np.prod(shape[1:])
                Nt = NtNxyz // Nxyz
                if not NtNxyz % Nxyz == 0:
                    raise ValueError(
                        f"Shape of incomplete data in '{self.filename}' inconsistent "
                        + f"with shape={shape}: Nt={NtNxyz / Nxyz} must be an integer."
                    )
                shape = (Nt,) + shape[1:]
            _data = _data.reshape(shape)
        else:
            raise NotImplementedError(
                f"Data format of '{self.filename}' not supported."
            )
        self._data = _data


@implementer(IWData)
class WData(object):
    """Base implementation."""

    # This is the extension used for infofiles
    _infofile_extension = "wtxt"

    # Allowed values for dx, dt, etc. for variable dimensions.  We use np.nan
    # internally.
    _varying = set([np.nan, None, NotImplemented, "varying"])

    def __init__(
        self,
        prefix="tmp",
        description="",
        data_dir=".",
        ext="wdat",
        dim=None,
        Nxyz=None,
        dxyz=(1, 1, 1),
        xyz0=None,
        xyz=None,
        Nt=0,
        t0=0,
        dt=1,
        t=None,
        variables=None,
        aliases=None,
        constants=None,
        check_data=True,
    ):
        self.prefix = prefix
        self.description = description
        self.data_dir = data_dir
        self.ext = ext
        self.variables = variables if variables else []
        self.aliases = aliases if aliases else {}
        self.constants = constants if constants else {}
        self._dim = dim
        self.xyz, self.Nxyz, self.dxyz, self.xyz0 = xyz, Nxyz, dxyz, xyz0
        self.t, self.Nt, self.dt, self.t0 = t, Nt, dt, t0
        self.check_data = check_data
        self.init()

    def init(self):
        """Perform all initialization and checks.

        Called by the constructor, and before saving.
        """
        # Abscissa
        if self.Nxyz is None:
            if self.xyz is None:
                raise ValueError("Must provide one of xyz or Nxyz")

            xyz = []
            for _x in self.xyz:
                if _x is not None:
                    _x = np.ravel(_x)
                xyz.append(_x)

            Nxyz, dxyz, xyz0 = [], [], []
            for x in xyz:
                Nx = None
                x0 = dx = np.nan
                if x is not None:
                    dxs = np.diff(x)
                    if len(dxs) > 0:
                        dx = dxs.mean()
                    if len(dxs) == 0 or not np.allclose(dxs, dx):
                        dx = np.nan
                    Nx = len(x)
                    x0 = x[0]

                Nxyz.append(Nx)
                dxyz.append(dx)
                xyz0.append(x0)
        else:
            xyz, Nxyz, dxyz, xyz0 = self.xyz, self.Nxyz, self.dxyz, self.xyz0
            if Nxyz[0] <= 3:
                raise ValueError(f"First dimension of Nxyz=={Nxyz} must be > 3.")

            if xyz is None:
                xyz = (None,) * len(Nxyz)

            if xyz0 is None:
                xyz0 = (None,) * len(Nxyz)

            # Allow for individual None values or NaN values.
            xyz0 = [
                -_N * _d / 2 if _x0 is None or np.isnan(_x0) else _x0
                for _N, _d, _x0 in zip(Nxyz, dxyz, xyz0)
            ]
            xyz = [
                np.arange(_N) * _dx + _x0 if _x is None else _x
                for _x, _N, _dx, _x0 in zip(xyz, Nxyz, dxyz, xyz0)
            ]

            # While Nxyz etc. can be longer than dim, if dim is specified, we must
            # truncate xyz so broadcasting works.
            if self._dim is not None:
                xyz = xyz[: self._dim]

        # Make sure abscissa are appropriately broadcast.
        self.xyz = np.meshgrid(*xyz, indexing="ij", sparse=True)
        self.Nxyz, self.dxyz, self.xyz0 = map(tuple, (Nxyz, dxyz, xyz0))

        # Times
        if self.t is not None:
            t = np.ravel(self.t)
            Nt = len(t)
            dts = np.diff(t)
            if len(dts) > 0:
                dt = dts.mean()
            if len(dts) == 0 or not np.allclose(dts, dt):
                dt = np.nan
            t0 = t[0]
        else:
            Nt, dt, t0 = self.Nt, self.dt, self.t0
            if Nt == 0:
                if self.variables is not None:
                    for var in self.variables:
                        if var.data is not None:
                            Nt = var.data.shape[0]
                            break

            t = np.arange(Nt) * dt + t0

        self.t, self.Nt, self.dt, self.t0 = t, Nt, dt, t0

        # Check variables
        if self.check_data and self.variables is not None:
            dim = self.dim
            Nt = self.Nt
            for var in self.variables:
                if var.data is not None:
                    name, data = var.name, var.data
                    if Nt != var.data.shape[0]:
                        raise ValueError(
                            f"Variable '{name}' has incompatible Nt={Nt}:"
                            + f" data.shape[0] = {data.shape[0]}"
                        )
                    if var.data.shape[-self.dim :] != self.Nxyz[: self.dim]:
                        raise ValueError(
                            f"Variable '{name}' has incompatible Nxyz={Nxyz}:"
                            + f" data.shape[-{self.dim}:] = {data.shape[-self.dim:]}"
                        )
                    if (var.vector and len(var.shape) - 2 != dim) or (
                        not var.vector and len(var.shape) - 1 != dim
                    ):
                        raise ValueError(
                            f"Variable '{name}' has incompatible dim={dim}:"
                            + f" data.shape = {data.shape}"
                        )

    @property
    def dim(self):
        dim = self._dim
        if dim is None:
            dim = len(self.xyz)
        return dim

    def _get_ext(self, var):
        """Return the extension of ``var``."""
        if var.filename:
            return var.filename.split(".")[-1]
        return self.ext

    def get_metadata(self, header=None):
        Nxyz = self.Nxyz
        dxyz = tuple(
            _dx if _dx not in self._varying else "varying" for _dx in self.dxyz
        )
        dt = self.dt if self.dt not in self._varying else "varying"

        descriptors = (
            [
                (f"N{_x}".lower(), _N, _l)
                for _x, _N, _l in zip(
                    "xyz",
                    Nxyz,
                    (
                        "Lattice size in x direction",
                        "            ... y ...",
                        "            ... z ...",
                    ),
                )
            ]
            + [
                (f"d{_x}".lower(), _N, _l)
                for _x, _N, _l in zip(
                    "xyz",
                    dxyz,
                    (
                        "Spacing in x direction",
                        "       ... y ...",
                        "       ... z ...",
                    ),
                )
            ]
            + [
                ("prefix", self.prefix, "datafile prefix: <prefix>_<var>.<format>"),
                ("datadim", self.dim, "Block size: 1:Nx, 2:Nx*Ny, 3:Nx*Ny*Nz"),
                ("cycles", self.Nt, "Number Nt of frames/cycles per dataset"),
                ("t0", self.t0, "Time value of first frame"),
                ("dt", dt, "Time interval between frames"),
            ]
        )

        # Add x0, y0, z0 if not default
        for x0, dx, Nx, x in zip(self.xyz0, self.dxyz, self.Nxyz, "xyz"):
            if dx not in self._varying and np.allclose(x0, -Nx * dx / 2):
                continue
            descriptors.append((f"{x}0", x0, f"First point in {x} lattice"))

        # Add comments here
        lines = []

        if header is not None:
            lines.extend(["# " + _v for _v in header.splitlines()])

        if self.description is not None:
            lines.extend(["# " + _v for _v in self.description.splitlines()])

        if lines:
            lines.extend([""])

        # Process text and pad for descriptors
        lines.extend(
            pad_and_justify(
                [
                    (_name, str(_value), "# " + _comment)
                    for (_name, _value, _comment) in descriptors
                ]
            )
        )

        # Add variables here
        if self.variables:
            lines.extend(["", "# variables"])
            lines.extend(
                pad_and_justify(
                    [("# tag", "name", "type", "unit", "format", "# description")]
                    + [
                        (
                            "var",
                            _v.name,
                            self._get_type(_v),
                            _v.unit,
                            self._get_ext(var=_v),
                            f"# {_v.description}",
                        )
                        for _v in self.variables
                    ]
                )
            )

        # Add aliases here
        if self.aliases:
            lines.extend(["", "# links"])
            lines.extend(
                pad_and_justify(
                    [("# tag", "name", "link-to")]
                    + [("link", _k, self.aliases[_k]) for _k in self.aliases]
                )
            )

        # Add constants
        if self.constants:
            lines.extend(["", "# consts"])
            lines.extend(
                pad_and_justify(
                    [("# tag", "name", "value")]
                    + [("const", _k, repr(self.constants[_k])) for _k in self.constants]
                )
            )
        metadata = "\n".join([_l.rstrip() for _l in "\n".join(lines).splitlines()])
        return metadata

    @property
    def infofile(self):
        return os.path.join(
            self.data_dir, ".".join([self.prefix, self._infofile_extension])
        )

    def save(self, force=False):
        t1, t2 = current_time()
        metadata = self.get_metadata(header=f"Generated by wdata.io: [{t1} = {t2}]")

        data_dir = self.data_dir
        if not os.path.exists(data_dir):
            if force:
                os.makedirs(data_dir, exist_ok=True)
            else:
                raise IOError(f"Directory data_dir={data_dir} does not exist.")

        infofile = self.infofile
        if os.path.exists(infofile) and not force:
            raise IOError(f"File '{infofile}' already exists!")

        with open(infofile, "w") as f:
            f.write(metadata)

        variables = list(self.variables)
        if self.xyz is not None:
            for _x, _dx, _n in zip(self.xyz, self.dxyz, "xyz"):
                if np.isnan(_dx):
                    # Add non-linearly varying abscissa
                    _name = f"_{_n}"  # Abscissa start with underscores
                    variables.append(Var(**{_name: np.ravel(_x)}))
        if self.t is not None and np.isnan(self.dt):
            variables.append(Var(_t=np.ravel(self.t)))

        for var in variables:
            if var._data is None:
                continue
            filename = os.path.join(data_dir, f"{self.prefix}_{var.name}.{self.ext}")
            var.write_data(filename=filename, force=force)

    @classmethod
    def load(cls, infofile=None, full_prefix=None, check_data=True, **kw):
        kw.update(check_data=check_data)
        if infofile is not None:
            # Load from infofile
            if full_prefix is None:
                return cls.load_from_infofile(infofile=infofile, **kw)
            raise ValueError(
                f"Got both infofile={infofile} and" + f" full_prefix={full_prefix}."
            )
        elif full_prefix is None:
            raise ValueError("Must provide either infofile or full_prefix.")

        # Check if full_prefix shows an infofile.
        infofile = ".".join([full_prefix, cls._infofile_extension])
        if os.path.exists(infofile):
            return cls.load_from_infofile(infofile=infofile, **kw)

        # No infofile option.
        raise NotImplementedError("infofile={infofile} must currently exist.")

    @classmethod
    def load_from_infofile(cls, infofile, **kw):
        """Load data from specified infofile."""
        with open(infofile, "r") as f:
            lines = f.readlines()

        data_dir = os.path.dirname(infofile)

        # Extract header - initial comments (skipping blank lines)
        header = []
        while lines:
            if not lines[0].strip():
                pass
            elif lines[0].startswith("#"):
                header.append(lines[0][1:].strip())
            else:  # pragma: nocover  https://github.com/nedbat/coveragepy/issues/772
                break
            lines.pop(0)

        description = "\n".join(header)

        # Pairs of [line, comment] or [line]
        lines = [_l.split("#", 1) for _l in lines]

        # Pairs of ([terms], [comment]) or ([terms], []) if no comment
        lines = [([_w.strip() for _w in _l[0].split()], _l[1:]) for _l in lines if _l]
        lines = [_l for _l in lines if _l[0]]  # Skip blank lines

        parameters = OrderedDict()
        variables = []
        aliases = OrderedDict()
        constants = OrderedDict()

        for line in lines:
            terms, comments = line
            if terms[0].lower() == "var":
                # tag  name     [type]   [unit]  [format]
                # var  density   real     none    npy
                terms.pop(0)
                name = terms.pop(0)
                type = terms.pop(0) if terms else None  # Optional
                unit = terms.pop(0) if terms else None
                ext = terms.pop(0) if terms else "npy"

                # Dummy shape for now.
                if type == "vector":
                    shape = (None, 3, 100)
                else:
                    shape = (None, 100)

                variables.append(
                    Var(
                        name=name,
                        unit=unit,
                        filename=ext,  # Until we know prefix...
                        shape=shape,
                        descr=cls._get_descr(type=type),
                        description=" ".join(comments),
                    )
                )
            elif terms[0].lower() == "link":
                # tag   name       link-to
                # link  density_b  density_a
                terms.pop(0)
                name = terms.pop(0)
                link = terms.pop(0)
                aliases[name] = link
            elif terms[0].lower() == "const":
                # tag    name  value  [unit]
                # const  kF    1.0     none
                terms.pop(0)
                name = terms.pop(0)
                value = terms.pop(0)
                # unit = terms.pop(0) if terms else None
                constants[name] = eval(value, constants)
            else:
                # param  value
                # nx     24
                name = terms.pop(0)
                value = terms.pop(0)
                name = name.lower()  # Parameter names are case-insensitive.
                parameters[name] = value

        # Process parameters
        if "prefix" not in parameters:
            prefix = os.path.basename(infofile)
            if prefix.endswith("." + cls._infofile_extension):
                prefix = prefix[: -1 - len(cls._infofile_extension)]
            else:
                prefix = prefix.rsplit(".", 1)[0]

            warn(f"No prefix specified in {infofile}: assuming prefix={prefix}")
            parameters["prefix"] = prefix

        prefix = parameters["prefix"]

        _xyz = "".join([_x for _x in "xyz" if f"n{_x}" in parameters])
        Nxyz = tuple(int(parameters.pop(f"n{_x}")) for _x in _xyz)
        dim = int(parameters.pop("datadim", len(Nxyz)))
        dxyz = tuple(
            cls._float(parameters.pop(f"d{_x}", "varying").lower()) for _x in _xyz
        )
        xyz0 = tuple(float(parameters.pop(f"{_x}0", np.nan)) for _x in _xyz)

        Nt = int(parameters.pop("cycles", 0))
        dt = cls._float(parameters.pop("dt", "varying").lower())

        # Load abscissa if any dx or dt are nan indicating unequal spacing.
        abscissa = {}
        for _x, _dx in zip("t" + _xyz, (dt,) + dxyz):
            if np.isnan(_dx):
                filename = os.path.join(data_dir, f"{prefix}__{_x}.*")
                files = sorted(glob.glob(filename))
                if len(files) == 0:
                    raise ValueError(
                        f"Abscissa {_x} has varying d{_x} but no files '{filename}' found."
                    )
                elif len(files) > 1:
                    warn(
                        f"Multiple files found for varying abscissa {_x}: {files}\n"
                        + f"Using {files[0]}"
                    )
                f = files[0]
                # ext = f.split(".")[-1]
                abscissa[_x] = Var(name=_x, shape=(None,), descr=float, filename=f).data

        xyz = tuple(abscissa.get(_x, None) for _x in _xyz)
        t = abscissa.get("t", None)

        # Add filenames and shapes.  Defer loading until the user needs the data
        for var in variables:
            ext = var.filename.split(".")[-1]
            var.filename = os.path.join(data_dir, f"{prefix}_{var.name}.{ext}")
            if var.vector:
                var.shape = (Nt, 3) + Nxyz[:dim]
            else:
                var.shape = (Nt,) + Nxyz[:dim]

        args = dict(
            prefix=prefix,
            description=description,
            data_dir=data_dir,
            dim=dim,
            Nxyz=Nxyz,
            dxyz=dxyz,
            xyz0=xyz0,
            xyz=xyz,
            Nt=Nt,
            t0=float(parameters.pop("t0", 0)),
            dt=dt,
            t=t,
            variables=variables,
            aliases=aliases,
            constants=constants,
        )
        args.update(kw)

        wdata = cls(**args)
        return wdata

    @classmethod
    def _float(cls, val):
        """Return float(value) defaulting to np.nan for varying values."""
        if val in cls._varying:
            return np.nan
        return float(val)

    def load_data(self, *names):  # pragma: nocover
        """Load the specified data."""
        raise NotImplementedError

    def keys(self):
        keys = set([_var.name for _var in self.variables])
        if self.aliases:
            keys.update(self.aliases)
        if self.constants:
            keys.update(self.constants)
        return sorted(keys)

    def __dir__(self):
        return self.keys()

    def __iter__(self):
        return self.keys().__iter__()

    def __len__(self):
        return len(self.keys())

    def __getattr__(self, key):
        if self.aliases:
            key = self.aliases.get(key, key)
        if self.variables:
            for var in self.variables:
                if var.name == key:
                    if key in self.constants:
                        warn(f"Variable {key} hides constant of the same name")
                    return var.data

        if self.constants and key in self.constants:
            return self.constants[key]

        return super().__getattribute__(key)

    ######################################################################
    ####################
    # Helpers
    @staticmethod
    def _get_type(var):
        """Return the type string for backward compatibility.

        Arguments
        ---------
        var : IVar
            Variable.
        """
        if len(var.shape) == 1:
            return "abscissa"
        elif var.vector:
            assert var.descr == "<f8"
            return "vector"

        descr = np.dtype(var.descr)

        if descr == "<c16":
            return "complex"
        elif descr == "<f8":
            return "real"
        else:
            assert len(descr.descr) == 1
            assert len(descr.descr[0]) == 2
            assert descr.descr[0][0] == ""
            return descr.descr[0][1]

    @staticmethod
    def _get_descr(type):
        """Return the descr from type string for backward compatibility.

        Arguments
        ---------
        type : str
           The type field in the original WDAT format.
        """
        if type in ("vector", "real"):
            return "<f8"

        if type == ("complex"):
            return "<c16"

        return type

    def __eq__(self, data):
        """Return True if the two datasets are equivalent in terms of data.

        This is used for testing.
        """
        if (
            self.keys() != data.keys()
            or self.constants != data.constants
            or self.Nxyz != data.Nxyz
            or not np.array_equal(self.t, data.t)
            or not any([np.array_equal(_A, _B) for _A, _B in zip(self.xyz, data.xyz)])
        ):
            return False
        for key in self.keys():
            # Iterate over times in case arrays are too big for memory
            for i, t in enumerate(self.t):
                if not np.array_equal(
                    getattr(self, key)[i], getattr(data, key)[i], equal_nan=True
                ):
                    return False
        return True


def load_wdata(prefix=None, infofile=None, **kw):
    if infofile is not None:
        return WData.load(infofile=infofile, **kw)
    else:
        raise NotImplementedError()


######################################################################
# Utilities and helpers
def current_time(format="%Y-%m-%d %H:%M:%S %Z%z"):
    """Return the date and time as strings (utc, local)."""
    now_utc = datetime.now(timezone("UTC"))
    # Convert to local time zone
    now_local = now_utc.astimezone(tzlocal.get_localzone())
    return (now_utc.strftime(format), now_local.strftime(format))


def pad_and_justify(lines, padding="    "):
    """Return list of padded aligned strings.

    First string is left-justified, the remaining are
    right-justified except comments.
    """
    # Add padding and justify
    justify = [str.ljust] + [str.rjust] * len(lines[0])
    for _line in lines:
        for _n in range(1, len(_line)):
            if "#" in _line[_n]:
                # Left-justify comment columns
                justify[_n] = str.ljust

    lens = np.max([[len(_v) for _v in _d] for _d in lines], axis=0)
    return [
        padding.join([justify[_n](_v, lens[_n]) for _n, _v in enumerate(_line)])
        for _line in lines
    ]
