# automatically generated by the FlatBuffers compiler, do not modify

# namespace: schema

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class RequestAsyncGetModel(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = RequestAsyncGetModel()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsRequestAsyncGetModel(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def RequestAsyncGetModelBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x46\x4C\x4A\x30", size_prefixed=size_prefixed)

    # RequestAsyncGetModel
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # RequestAsyncGetModel
    def FlName(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # RequestAsyncGetModel
    def Iteration(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

def Start(builder): builder.StartObject(2)
def RequestAsyncGetModelStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddFlName(builder, flName): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(flName), 0)
def RequestAsyncGetModelAddFlName(builder, flName):
    """This method is deprecated. Please switch to AddFlName."""
    return AddFlName(builder, flName)
def AddIteration(builder, iteration): builder.PrependInt32Slot(1, iteration, 0)
def RequestAsyncGetModelAddIteration(builder, iteration):
    """This method is deprecated. Please switch to AddIteration."""
    return AddIteration(builder, iteration)
def End(builder): return builder.EndObject()
def RequestAsyncGetModelEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)