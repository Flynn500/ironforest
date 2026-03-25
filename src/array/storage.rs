use std::sync::Arc;
use pyo3::{Py, PyAny};
use pyo3::buffer::PyBuffer;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

pub enum Storage<T> {
    Owned(Vec<T>),

    External {
        owner: Py<PyAny>,
        ptr: *const T,
        len: usize,
    },

    Strided {
        base: Arc<[T]>,
        offset: usize,
        len: usize,
    },

    Buffer {
        buf: PyBuffer<T>,
        len: usize,
    },
}

unsafe impl<T: Send> Send for Storage<T> {}
unsafe impl<T: Sync> Sync for Storage<T> {}

impl<T> Storage<T> {
    pub fn from_vec(data: Vec<T>) -> Self {
        Storage::Owned(data)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Storage::Owned(Vec::with_capacity(capacity))
    }

    pub fn into_vec(self) -> Vec<T> {
        match self {
            Storage::Owned(v) => v,
            Storage::External { .. } => {
                panic!("into_vec() called on read-only External storage; \
                        clone the NdArray first to get an Owned copy")
            }
            Storage::Strided { .. } => {
                panic!("into_vec() called on Strided storage; \
                        call to_contiguous() on the NdArray first")
            }
            Storage::Buffer { .. } => {
                panic!("into_vec() called on read-only Buffer storage; \
                        clone the NdArray first to get an Owned copy")
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Storage::Owned(v) => v.len(),
            Storage::External { len, .. } => *len,
            Storage::Strided { len, .. } => *len,
            Storage::Buffer { len, .. } => *len,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn is_contiguous(&self) -> bool {
        !matches!(self, Storage::Strided { .. })
    }

    pub fn is_owned(&self) -> bool {
        matches!(self, Storage::Owned(_))
    }

    pub fn is_buffer(&self) -> bool {
        matches!(self, Storage::Buffer { .. })
    }

    pub fn as_slice(&self) -> Option<&[T]> {
        match self {
            Storage::Owned(v) => Some(v.as_slice()),
            Storage::External { ptr, len, .. } => Some(unsafe {
                std::slice::from_raw_parts(*ptr, *len)
            }),
            Storage::Strided { .. } => None,
            Storage::Buffer { buf, len } => Some(unsafe {
                std::slice::from_raw_parts(buf.buf_ptr() as *const T, *len)
            }),
        }
    }

    #[inline]
    pub fn as_slice_unchecked(&self) -> &[T] {
        self.as_slice()
            .expect("as_slice_unchecked() called on Strided storage; call to_contiguous() first")
    }

    pub fn as_raw_ptr(&self) -> *const T {
        match self {
            Storage::Owned(v) => v.as_ptr(),
            Storage::External { ptr, .. } => *ptr,
            Storage::Strided { base, offset, .. } => unsafe { base.as_ptr().add(*offset) },
            Storage::Buffer { buf, .. } => buf.buf_ptr() as *const T,
        }
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        match self {
            Storage::Owned(v) => Some(v.as_mut_slice()),
            Storage::External { .. } | Storage::Strided { .. } | Storage::Buffer { .. } => None,
        }
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        match self {
            Storage::Owned(v) => v.get(index),
            Storage::External { ptr, len, .. } => {
                if index < *len {
                    Some(unsafe { &*ptr.add(index) })
                } else {
                    None
                }
            }
            Storage::Strided { base, offset, len } => {
                let actual = offset + index;
                if index < *len && actual < base.len() {
                    Some(&base[actual])
                } else {
                    None
                }
            }
            Storage::Buffer { buf, len } => {
                if index < *len {
                    Some(unsafe { &*(buf.buf_ptr() as *const T).add(index) })
                } else {
                    None
                }
            }
        }
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        match self {
            Storage::Owned(v) => v.get_mut(index),
            Storage::External { .. } | Storage::Strided { .. } | Storage::Buffer { .. } => None,
        }
    }
}

impl<T: Clone> Storage<T> {
    pub fn filled(value: T, len: usize) -> Self {
        Storage::Owned(vec![value; len])
    }

    pub fn to_owned_vec(&self) -> Vec<T> {
        match self {
            Storage::Owned(v) => v.clone(),
            Storage::External { ptr, len, .. } => unsafe {
                std::slice::from_raw_parts(*ptr, *len).to_vec()
            },
            Storage::Strided { base, offset, len } => {
                base[*offset..*offset + *len].to_vec()
            }

            Storage::Buffer { buf, len } => unsafe {
                std::slice::from_raw_parts(buf.buf_ptr() as *const T, *len).to_vec()
            },
        }
    }
}

impl<T: Default + Clone> Storage<T> {
    pub fn zeros(len: usize) -> Self {
        Storage::Owned(vec![T::default(); len])
    }
}


impl<T: std::fmt::Debug> std::fmt::Debug for Storage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Storage::Owned(v) => f.debug_tuple("Owned").field(v).finish(),
            Storage::External { len, .. } => {
                f.debug_struct("External").field("len", len).finish()
            }
            Storage::Strided { offset, len, .. } => f
                .debug_struct("Strided")
                .field("offset", offset)
                .field("len", len)
                .finish(),
            Storage::Buffer { len, .. } => {
                f.debug_struct("Buffer").field("len", len).finish()
            }
        }
    }
}

impl<T: Clone> Clone for Storage<T> {
    fn clone(&self) -> Self {
        Storage::Owned(self.to_owned_vec())
    }
}

impl<T: Serialize + Clone> Serialize for Storage<T> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Storage::Owned(v) => v.serialize(serializer),
            _ => self.to_owned_vec().serialize(serializer),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Storage<T> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let v = Vec::<T>::deserialize(deserializer)?;
        Ok(Storage::Owned(v))
    }
}
