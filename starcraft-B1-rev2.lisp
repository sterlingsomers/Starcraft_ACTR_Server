;;; Version Notes: renamed to starcraft from starcrat
;;; A1rev3: Added a second production that /should/ repond to imaginal bufferings.  It doesn't fire yet.
;;; A1rev2: Modified to receive a response
;;; - reduced the number of cycles to 1000
;;; A1rev1: just a loop model. Does not receive a response


;;; Copyright (c) 2017 Carnegie Mellon University
;;;
;;; Permission is hereby granted, free of charge, to any person obtaining a copy of this
;;; software and associated documentation files (the "Software"), to deal in the Software
;;; without restriction, including without limitation the rights to use, copy, modify,
;;; merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
;;; permit persons to whom the Software is furnished to do so, subject to the following
;;; conditions:
;;;
;;; The above copyright notice and this permission notice shall be included in all copies
;;; or substantial portions of the Software.
;;;
;;; THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
;;; INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
;;; PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
;;; HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
;;; CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
;;; OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

;;; This is a trivial example of Voorhees in action. It uses the addition model from the
;;; first unit of the ACT-R tutorial to add numbers, the numbers being supply over a
;;; TCP connection, and the result then being returned over that connection.
;;;
;;; To run the example first ensure you have Voorhees installed as suggested in its
;;; documentation, so that it can be loaded with QuickLisp. Then cd to a directory that
;;; contains an actr7 directory. (Alternatively you can set *actr7-parent*, below, to such
;;; a directory and run this from anywhere.) Then launch Lisp and load this file. It will
;;; start listenting for connections on port 9907 (you can change this port below). Then,
;;; from a separate terminal, send it a JSON object such as
;;;
;;; { "arg1": 3, "arg2": 5 }
;;;
;;; The ACT-R model will be run and the sum returned. For example, if you use netcat (nc)
;;; to connection the interaction in the second terminal might look something like this:
;;;
;;; $ nc localhost 9907
;;; {"arg1": 3, "arg2": 5}
;;; 8
;;; {"arg1": 3, "arg2": 2}
;;; 5
;;;



;; Define the ACT-R model. This will generate a pair of warnings about slots in the goal
;; buffer not being accessed from productions; these may be ignored.

(clear-all)

(define-model sc2-model

(sgp :esc t :lf .05 :v t)
;;lat = 38.967163
;;lon = -104.81937
;;t_lat = 38.99343742021595
;;t_lon = -105.05530266366
(chunk-type initialize state)
(chunk-type initial-info my_lat my_lon tar_lat tar_lon state)


(add-dm
 (goal ISA initialize state select-army)
 (start-location ISA waypoint-location my_lat 38.967163 my_lon -104.81937 tar_lat 38.993437 tar_lon -105.055303 state 1))

(P clear-mission
   =goal>
       ISA        initialize
       state      select-army
   =imaginal>
     - wait       true
   ==>
   =goal>
       state      check-neutrals
   =imaginal>
       wait       true
   
   !eval! ("set_response" "_SELECT_ARMY" "[_SELECT_ALL]")
   ;!eval! (format t "msg")
)



(P wait
   =imaginal>
       wait       true
   ==>
   =imaginal>
   
)
;;should the wait production tic as well?
;;my initial thought is no because it 
;;should be happening between tics


(P check-neutrals
   =goal>
       ISA        initialize
       state      check-neutrals
   =imaginal>
       neutral_x  =nx
       neutral_y  =ny
    -  wait       true
   ==>
   =goal>
;       state      none
   =imaginal>
       wait       true
   ;-imaginal>

   !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" =nx =ny )
)


     

(goal-focus goal)
) ; end define-model

