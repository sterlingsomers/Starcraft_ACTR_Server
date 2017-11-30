;;; Version Notes:
;;; A1rev2: Modified to receive a response
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

;; Run this in the CL-USER package 'cause that's the easiest place to use ACT-R.
(in-package :cl-user)

;; A couple of default values that can be changed if necessary.
(defparameter *actr7-parent* nil)

(defparameter *port* 14555)

(defparameter *msg* nil)

(defparameter *map* nil)

(defparameter *tic* nil)

;; Load ACT-R if it's not already present, and then Voorhees.
#-ACT-R
(load (merge-pathnames (make-pathname :directory "actr7"
                                      :name "load-act-r"
                                      :type "lisp")
                       *actr7-parent*)
      :verbose t
      :print nil)

(ql:quickload :voorhees)

;; Define the ACT-R model. This will generate a pair of warnings about slots in the goal
;; buffer not being accessed from productions; these may be ignored.

(clear-all)

(define-model plan-model

(sgp :esc t :lf .05)
;;lat = 38.967163
;;lon = -104.81937
;;t_lat = 38.99343742021595
;;t_lon = -105.05530266366
(chunk-type initialize state)
(chunk-type initial-info my_lat my_lon tar_lat tar_lon state)


(add-dm
 (goal ISA initialize state clear-mission)
 (start-location ISA waypoint-location my_lat 38.967163 my_lon -104.81937 tar_lat 38.993437 tar_lon -105.055303 state 1))

(P clear-mission
   =goal>
       ISA        initialize
       state      clear-mission
   ==>
   =goal>
       state      clear-mission
   !eval! (setf *msg* (list (cons "get_observation" 32)))
)





;;{"Map_Request": {"args": {"lat": 38.967163, "meters": 7300, "lon": -104.81937, "exclude_list": []}, "method": "pathInRadius"}}
       ;;!eval! (setf *msg*


;;!eval! (setf *msg* `((lat .,=lat)(lon ., =lon)(meth . "foo")





(goal-focus goal)
) ; end define-model


;; Define the function that will do the work.
(defun run-plan-model (args)
  "Creates a chunk from the parsed JSON args, sets it to be the goal, runs the model,
and then returns the value of the sum slot of the chunk in the goal buffer. Note that an
integer is itself a JSON value."
  (let (socket)
    (unwind-protect
        (let ((stream (usocket:socket-stream
                       (setf socket (usocket:socket-connect "127.0.0.1" 33333 :timeout 100)))))
          (loop for x from 1 to 8000
            do (run 0.05)
            do (setf  *tic* (list (cons "command" "tic")))
            ;;do (format t "here-1")
            do (vh:write-json *tic* stream)
            do (format t "here-0.5")
            ;;do (let ((result (vh:read-json stream))))
            ;;do (format t "here0")
            do (if *msg*
                   (progn
                     ;;(format t (vh:json-string *msg*))
                     
                     (vh:write-json *msg* stream)
                     (format t "here")
                     (let ((result (vh:read-json stream)))
                       (unless result
                         (setf result (vh:read-json stream)))
                       (format t "here2")
                       (unless (assoc 'ack result)
                         (vh:chunkify result :buffer 'imaginal)))
                     (format t "here3")
                     ))
                     ;(let (result (read-line stream nil)) (format t "asdf"))))
                     ;(let (result (vh::unpack-json (st-json:read-json-from-string (read-line stream nil))))
                     ;  (format t "test"))))
            do (setf *msg* nil))

          (when socket
            (usocket:socket-close socket))))))
  ;(goal-focus-fct (vh:chunkify args))
  ;(run 0.5))
  ;(no-output (chunk-slot-value-fct (first (buffer-chunk goal)) 'sum)))


;; Run Voorhees and wait for connections, writing log information to the terminal, where
;; it will be interspersed with what ACT-R spits out.
;;(vh:run-model #'run-addition-model *port*
;;              :log-file *terminal-io*
;;              :log-json t)
