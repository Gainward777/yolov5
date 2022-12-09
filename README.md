<strong>The changes made to this copy of YOLOv5 are intended primarily to get rid of post-processing on the device after export.
<br>
Additionally, a simple sorting of results was added to work with ordered recognized objects, such as numerical sequences.</strong>
<br><br>
 ## <div align="center">Changelog:</div>
<br>
<ul type="disc">
 <li><b>models/experimental.py</b>
  <br>   
  line 118 add:  
  <pre>#sort ordered recognized objects
  def new_sorter</pre>   
  line 126 add:
  <pre>#nms from main model with little changes that solve the problem with the occurrence of errors when exporting using torch.jit.trace
  def nms_lite</pre> 
 </li> 
 <li><b> models/yolo.py</b>
  <br>
  line 169 add:  
  <pre>#flag for add postprocessing to export. when True on device u got only detections
  is_export=False</pre>  
  line 172 add:  
  <pre>#defoult treshhold for experimental.py/new_sort (sort detected digits on axis X)
  treshhold=0.8</pre>  
  line 214 (def forward) change:
  <pre> if augment to if augment and not self.is_export </pre>
  line 216 change:
  <pre>return self._forward_once(x, profile, visualize)</pre>
  to (our outputs)
  <pre>
  out=self._forward_once(x, profile, visualize)    
  if self.is_export:      
    return new_sorter(nms_lite(out)[0], self.treshhold)      
  else:
    return out</pre>
 </li>
 <li><b>export.py</b>
  <br>
  line 120 change:
  <pre>f = file.with_suffix('.torchscript')</pre>
  to 
  <pre>f = file.with_suffix('.torchscript.ptl')</pre>
  line 542 add:
  <pre>model.is_export=True</pre>
 </li>
</ul>
