#[allow(dead_code)]
#[repr(C)]
pub enum Error {
    Success = 0,
    InvalidValue = 1,
    OutOfMemory = 2,
    NotInitialized = 3,
    Deinitialized = 4,
    ProfilerDisabled = 5,
    ProfilerNotInitialized = 6,
    ProfilerAlreadyStarted = 7,
    ProfilerAlreadyStopped = 8,
    NoDevice = 100,
    InvalidDevice = 101,
    InvalidImage = 200,
    InvalidContext = 201,
    ContextAlreadyCurrent = 202,
    MapFailed = 205,
    UnmapFailed = 206,
    ArrayIsMapped = 207,
    AlreadyMapped = 208,
    NoBinaryForGpu = 209,
    AlreadyAcquired = 210,
    NotMapped = 211,
    NotMappedAsArray = 212,
    NotMappedAsPointer = 213,
    EccUncorrectable = 214,
    UnsupportedLimit = 215,
    ContextAlreadyInUse = 216,
    PeerAccessUnsupported = 217,
    InvalidPtx = 218,
    InvalidGraphicsContext = 219,
    InvalidSource = 300,
    FileNotFound = 301,
    SharedObjectSymbolNotFound = 302,
    SharedObjectInitFailed = 303,
    OperatingSystem = 304,
    InvalidHandle = 400,
    NotFound = 500,
    NotReady = 600,
    IllegalAddress = 700,
    LaunchOutOfResources = 701,
    LaunchTimeout = 702,
    LaunchIncompatibleTexturing = 703,
    PeerAccessAlreadyEnabled = 704,
    PeerAccessNotEnabled = 705,
    PrimaryContextActive = 708,
    ContextIsDestroyed = 709,
    Assert = 710,
    TooManyPeers = 711,
    HostMemoryAlreadyRegistered = 712,
    HostMemoryNotRegistered = 713,
    HardwareStackError = 714,
    IllegalInstruction = 715,
    MisalignedAddress = 716,
    InvalidAddressSpace = 717,
    InvalidPc = 718,
    LaunchFailed = 719,
    NotPermitted = 800,
    NotSupported = 801,
    Unknown = 999
}

#[allow(dead_code)]
#[repr(C)]
pub enum MemcpyKind {
    HostToHost = 0,
    HostToDevice = 1,
    DeviceToHost = 2,
    DeviceToDevice = 3,
    Default = 4
}

extern "C" {
    pub fn cudaMalloc(devPtr: *mut *mut ::libc::c_void,
                      size: ::libc::c_ulong)
        -> Error;

    pub fn cudaMemcpy(dst: *mut ::libc::c_void,
                      src: *const ::libc::c_void,
                      count: ::libc::c_ulong,
                      kind: MemcpyKind)
        -> Error;
}
