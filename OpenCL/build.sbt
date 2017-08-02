val lwjglNatives: String = {
  if (util.Properties.isMac) {
    "natives-macos"
  } else if (util.Properties.osName.startsWith("Linux")) {
    "natives-linux"
  } else if (util.Properties.isWin) {
    "natives-windows"
  } else {
    throw new MessageOnlyException(s"lwjgl does not support ${util.Properties.osName}")
  }
}

libraryDependencies += "org.lwjgl" % "lwjgl-opencl" % "3.1.2"

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.2"

libraryDependencies += "org.lwjgl" % "lwjgl" % "3.1.2" % Runtime classifier lwjglNatives

libraryDependencies += "com.thoughtworks.future" %% "future" % "1.0.1"

fork := true
