# P8e: compile every generated macro liberty (<inst>/<inst>_<corner>_syn.lib under the current dir) to .db next to it.
# Run by sram/mc_gen.sh gen in lc_shell.  Prints DB_OK / DB_FAIL <file> <library> per file and an area line.
foreach lib [lsort [glob -nocomplain */*_syn.lib]] {
    set before [get_object_name [get_libs -quiet *]]
    if {[catch {read_lib $lib} msg]} { puts "DB_FAIL $lib read: $msg"; continue }
    set new {}
    foreach l [get_object_name [get_libs -quiet *]] { if {[lsearch -exact $before $l] < 0} { lappend new $l } }
    if {[llength $new] != 1} { puts "DB_FAIL $lib libraries: $new"; continue }
    set db [file rootname $lib].db
    if {[catch {write_lib $new -format db -output $db} msg] || ![file exists $db]} { puts "DB_FAIL $lib write: $msg"; continue }
    puts "DB_OK $lib $new"
    remove_lib $new
}
exit
