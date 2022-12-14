# automatically generated by the FlatBuffers compiler, do not modify

# namespace: schema

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReturnAllClientListSign(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReturnAllClientListSign()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReturnAllClientListSign(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # ReturnAllClientListSign
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReturnAllClientListSign
    def Retcode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ReturnAllClientListSign
    def Reason(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ReturnAllClientListSign
    def ClientListSign(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from mindspore_fl.schema.ClientListSign import ClientListSign
            obj = ClientListSign()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # ReturnAllClientListSign
    def ClientListSignLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ReturnAllClientListSign
    def ClientListSignIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # ReturnAllClientListSign
    def Iteration(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ReturnAllClientListSign
    def NextReqTime(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def Start(builder): builder.StartObject(5)
def ReturnAllClientListSignStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddRetcode(builder, retcode): builder.PrependInt32Slot(0, retcode, 0)
def ReturnAllClientListSignAddRetcode(builder, retcode):
    """This method is deprecated. Please switch to AddRetcode."""
    return AddRetcode(builder, retcode)
def AddReason(builder, reason): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(reason), 0)
def ReturnAllClientListSignAddReason(builder, reason):
    """This method is deprecated. Please switch to AddReason."""
    return AddReason(builder, reason)
def AddClientListSign(builder, clientListSign): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(clientListSign), 0)
def ReturnAllClientListSignAddClientListSign(builder, clientListSign):
    """This method is deprecated. Please switch to AddClientListSign."""
    return AddClientListSign(builder, clientListSign)
def StartClientListSignVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ReturnAllClientListSignStartClientListSignVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartClientListSignVector(builder, numElems)
def AddIteration(builder, iteration): builder.PrependInt32Slot(3, iteration, 0)
def ReturnAllClientListSignAddIteration(builder, iteration):
    """This method is deprecated. Please switch to AddIteration."""
    return AddIteration(builder, iteration)
def AddNextReqTime(builder, nextReqTime): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(nextReqTime), 0)
def ReturnAllClientListSignAddNextReqTime(builder, nextReqTime):
    """This method is deprecated. Please switch to AddNextReqTime."""
    return AddNextReqTime(builder, nextReqTime)
def End(builder): return builder.EndObject()
def ReturnAllClientListSignEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)