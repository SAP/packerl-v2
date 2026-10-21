# Create a custom library containing all source files except the main executable file
add_library(scratch-packerl-lib
        applications/disposable-application.cc
        applications/disposable-bulk-send-application.cc
        applications/disposable-onoff-application.cc
        monitoring/global-rewarder.cc
        monitoring/monitoring-node.cc
        monitoring/monitoring-edge.cc
        monitoring/node-rewarder.cc
        monitoring/packerl-monitor.cc
        odd-routing/odd-label-map.cc
        odd-routing/odd-node.cc
        odd-routing/odd-routing.cc
        odd-routing/odd-routing-helper.cc
        packerl.cc
        )

# Link the custom library to the scratch executable
build_exec(
  EXECNAME scratch-packerl-executable
  SOURCE_FILES sim.cc
  LIBRARIES_TO_LINK scratch-packerl-lib "${ns3-libs}" "${ns3-contrib-libs}"
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/packerl
)
