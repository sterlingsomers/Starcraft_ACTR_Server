;;; Version Notes: renamed to starcraft from starcrat
;;; A1rev3: Added a second production that /should/ repond to imaginal bufferings.  It doesn't fire yet.
;;; A1rev2: Modified to receive a response
;;; - reduced the number of cycles to 1000
;;; A1rev1: just a loop model. Does not receive a response




;; Define the ACT-R model. This will generate a pair of warnings about slots in the goal
;; buffer not being accessed from productions; these may be ignored.

(clear-all)

(define-model sc2-model

(sgp :esc t :lf .05 :v t)

(chunk-type initialize state)


(add-dm
 (goal ISA initialize state select-army))


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

   !eval! ("RHSWait")
   
)
;;should the wait production tic as well?
;;my initial thought is no because it 
;;should be happening between tics


(P click-beacon
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

