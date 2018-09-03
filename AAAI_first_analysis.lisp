;;; Version Notes: renamed to starcraft from starcrat
;;; A1rev3: Added a second production that /should/ repond to imaginal bufferings.  It doesn't fire yet.
;;; A1rev2: Modified to receive a response
;;; - reduced the number of cycles to 1000
;;; A1rev1: just a loop model. Does not receive a response




;; Define the ACT-R model. This will generate a pair of warnings about slots in the goal
;; buffer not being accessed from productions; these may be ignored.

(clear-all)
(require-extra "blending")
(define-model sc2-model


(sgp :esc t    ;;set to t to use subsymbolic computations
     :sim-hook "cosine-similarity" 
     ;:cache-sim-hook-results t
     ;:bln t    ;;added this to ensure blending was enabled, but not sure if it is actually doing anything...
     ;;some parameters moved from +actr-parameters+ in swarm model
     :ans nil;0.25 ;;activation noise, default is nil, swarm was .75 (M-turkers), Christian recommended .25 as a start.
     :tmp 1.0  ;;decouple noise if not nil, swarm was .7, perhaps change later.
     :mp 1   ;;partial matching enabled, default is off/nil, start high (2.5 is really high) and move to 1.
     :bll nil;0.5  ;;base-level learning enabled, default is off, this is the recommended value.
     :rt -10.0 ;;retrieval threshold, default is 0, think about editing this number.
     :blc 0;5    ;;base-level learning constant, default is 0, think about editing this number.
     :lf 0.25  ;;latency factor for retrieval times, default is 1.0, think about editing this number.
     :ol nil   ;;affects which base-level learning equation is used, default is nil; use 1 later maybe
     :md -2.5  ;;maximum difference, default similarity value between chunks, default is 1.0, maybe edit this.
     :ga 0.0   ;;set goal activation to 0.0 for spreading activation
     ;:imaginal-activation 1.0 ;;set imaginal activation to 0.0 for spreading activation
     ;:mas 1.0 ;;spreading activation
     ;;back to some parameters in the original sgp here
     :v t      ;;set to nil to turn off model output
     :blt t    ;;set to nil to turn off blending trace
     :trace-detail high ;;lower this as needed, start at high for initial debugging.
     :style-warnings t  ;;set to nil to turn off production warnings, start at t for initial debugging.
     :seed (1 1)
     :esc t
     
     ) ;;end sgp


(chunk-type initialize state)
(chunk-type decision green orange blocking action)
(chunk-type observation green orange blocking)
;(chunk-type action value)
;(chunk-type game-state value)

(add-dm
 (goal isa initialize state select-army))
 ;(select-orange isa action value select-orange)
 ;(select-green isa action value select-green)
 ;(select-beacon isa action value select-beacon)
 ;(select-around isa action value select-around))
;;chunks defined in Python and vector have random elements

(P clear-mission
   =goal>
       isa        initialize
       state      select-army
   =imaginal>
     - wait       true
   ?imaginal>
       state      free
   ==>
   =goal>
       state      check-neutrals
   =imaginal>
       green      nil
       orange     nil
       blocking    nil
       ;vector     nil
       ;value_estimate nil
       wait       true
   
   !eval! ("set_response" "_SELECT_ARMY" "[_SELECT_ALL]")
   ;!eval! (format t "msg")
)



;(P wait
;   =imaginal>
;       wait       true
;   ?imaginal>
;       state      free
;   ==>
;   =imaginal>
;       green      nil
;       orange     nil
;       blocking   nil
;       ;vector     nil
;       ;value_estimate nil
;       wait       true
;
;   !eval! ("RHSWait")
;   
;)
;;should the wait production tic as well?
;;my initial thought is no because it 
;;should be happening between tics


;(P click-beacon
;   =goal>
;       ISA        initialize
;       state      check-neutrals
;   =imaginal>
;       neutral_x  =nx
;       neutral_y  =ny
;    -  wait       true
;   ==>
;   =goal>
;;       state      none
;   =imaginal>
;       wait       true
;   ;-imaginal>
;
;   !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" =nx =ny )
;)

(P click-part-two
   =goal>
       isa        initialize
       state      check-neutrals
   =imaginal>
       isa        observation
       green      =green
       orange     =orange
       blocking    =between
       ;vector     =vector
       ;value_estimate =value_estimate
     - wait       true
   ?blending>
       state      free
==>
   +blending>
       isa        decision ;notice the change from game-state to decision
       green      =green
       orange     =orange
       blocking    =between
       ;vector     =vector
       ;value_estimate =value_estimate
       :ignore-slots (wait)
       ;:do-not-generalize (green orange between vector value_estimate)
   =imaginal>
       green      nil
       orange     nil
       blocking    nil
       ;vector     nil
       ;value_estimate nil
       wait       false
   =goal>
       state      get_action

  
)

(P get_action
   
   =goal>
       isa        initialize
       state      get_action
   =blending>
       isa        decision
       green      =green
       orange     =orange
       blocking    =between
       ;vector     =vector
       ;value_estimate =value_estimate
       action     =action
   =imaginal>
   ?blending>
       state      free
==>
  
   =goal>
       state      do_action
   @blending>
   @imaginal>
       action     =action
       wait       false

   !eval! ("do_salience")
   ;!eval! ("Blend")

)

;(P select-green
;   =goal>
;       state      do_action
;   =imaginal>
;       action     select-green
;     - wait       true
;
;==>
;    =goal>
;       state      check-neutrals
;    =imaginal>
;       wait       true
;       action     nil
;    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_green")
;)

;(P select-orange
;   =goal>
;       state      do_action
;   =imaginal>
;       action     select-orange
;     - wait       true
;==>
;    =goal>
;       state      check-neutrals
;    =imaginal>
;       wait       true
;       action     nil
;    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_orange")
;)

(P select-green
   =goal>
       state      do_action
   =imaginal>
     <=  action     0.5
     - wait       true
==>
    =imaginal>
       wait       true
       action     nil
    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_beacon")
)
(P select-orange
   =goal>
       state      do_action
   =imaginal>
     > action     0.5
     - wait       true
==>
    =imaginal>
      wait       true
       action     nil
    !eval! ("set_response" "_MOVE_SCREEN" "_NOT_QUEUED" 1 1 "select_around")
)
     

(goal-focus goal)
) ; end define-model

