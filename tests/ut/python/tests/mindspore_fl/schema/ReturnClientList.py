# automatically generated by the FlatBuffers compiler, do not modify

# namespace: schema

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class ReturnClientList(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = ReturnClientList()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsReturnClientList(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    # ReturnClientList
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # ReturnClientList
    def Retcode(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ReturnClientList
    def Reason(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # ReturnClientList
    def Clients(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # ReturnClientList
    def ClientsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # ReturnClientList
    def ClientsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        return o == 0

    # ReturnClientList
    def Iteration(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # ReturnClientList
    def NextReqTime(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

def Start(builder): builder.StartObject(5)
def ReturnClientListStart(builder):
    """This method is deprecated. Please switch to Start."""
    return Start(builder)
def AddRetcode(builder, retcode): builder.PrependInt32Slot(0, retcode, 0)
def ReturnClientListAddRetcode(builder, retcode):
    """This method is deprecated. Please switch to AddRetcode."""
    return AddRetcode(builder, retcode)
def AddReason(builder, reason): builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(reason), 0)
def ReturnClientListAddReason(builder, reason):
    """This method is deprecated. Please switch to AddReason."""
    return AddReason(builder, reason)
def AddClients(builder, clients): builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(clients), 0)
def ReturnClientListAddClients(builder, clients):
    """This method is deprecated. Please switch to AddClients."""
    return AddClients(builder, clients)
def StartClientsVector(builder, numElems): return builder.StartVector(4, numElems, 4)
def ReturnClientListStartClientsVector(builder, numElems):
    """This method is deprecated. Please switch to Start."""
    return StartClientsVector(builder, numElems)
def AddIteration(builder, iteration): builder.PrependInt32Slot(3, iteration, 0)
def ReturnClientListAddIteration(builder, iteration):
    """This method is deprecated. Please switch to AddIteration."""
    return AddIteration(builder, iteration)
def AddNextReqTime(builder, nextReqTime): builder.PrependUOffsetTRelativeSlot(4, flatbuffers.number_types.UOffsetTFlags.py_type(nextReqTime), 0)
def ReturnClientListAddNextReqTime(builder, nextReqTime):
    """This method is deprecated. Please switch to AddNextReqTime."""
    return AddNextReqTime(builder, nextReqTime)
def End(builder): return builder.EndObject()
def ReturnClientListEnd(builder):
    """This method is deprecated. Please switch to End."""
    return End(builder)