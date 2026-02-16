;;;; microgpt.lisp

(in-package #:microgpt)

(defun load-names ()
  (let ((path (asdf:system-relative-pathname :microgpt "input.txt"))
        (names (make-array 0 :element-type 'string
                             :adjustable t
                             :fill-pointer 0)))
    (with-open-file (stream path :direction :input)
      (loop for line = (read-line stream nil nil)
            while line
            do (vector-push-extend line names)))
    names))

(defun shuffle (vec)
  "Shuffle vector VEC in place using Fisher-Yates."
  (loop for i from (1- (length vec)) downto 1
        do (rotatef (aref vec i) (aref vec (random (1+ i)))))
  vec)

(defun main ()
  (let ((names (shuffle (load-names))))
    (format t "~a~%" (length names))))
