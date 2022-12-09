The changes made to this copy of YOLOv5 are intended primarily to get rid of post-processing on the device after export.
<br>
Additionally, a simple sorting of results was added to work with ordered recognized objects, such as numerical sequences.
<br>
<br>
<div align="center">Changelog:</div>
<br>
<br>
models/experimental.py
<br>
 #sort ordered recognized objects
<br>
str 117 add: def new_sorter,
<br>
str 125 add:
<br>
#nms from main model with little changes that solve the problem with the occurrence of errors when exporting using torch.jit.trace
<br>
def nms_lite 
<br>
<br>
models/yolo.py
<br>
<br>
str 169 add: 
<br>
is_export=False,
<br>
str 172 add: 
<br>
treshhold=0.8,
<br>
str 214 (def forward) change: 
<br>
if augment to if augment and not self.is_export,
<br>
str 216 change: 
<br>
return self._forward_once(x, profile, visualize) to
<br>
      out=self._forward_once(x, profile, visualize)
      <br>
      if self.is_export:
      <br>
          return new_sorter(nms_lite(out)[0], self.treshhold)
      <br>
      else:
      <br>
          return out
      <br>
export.py
<br>
<br>
str 120 change: 
<br>
f = file.with_suffix('.torchscript') to f = file.with_suffix('.torchscript.ptl'),
<br>
str 542 add: 
<br>
model.is_export=True,
