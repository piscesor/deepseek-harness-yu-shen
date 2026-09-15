/** Application and notification-area icon assets owned by the Electron shell. */

import { readFile } from 'node:fs/promises'
import { resolve } from 'node:path'
import { app, nativeImage, type NativeImage } from 'electron'

/**
 * Resolve one icon asset shipped beside the Electron application.
 * @param name - File name inside the application `assets` directory.
 * @returns Absolute path of the asset in the packaged or development application.
 */
export function desktopAssetPath(name: string): string {
  return resolve(app.getAppPath(), 'assets', name)
}

/**
 * Read one icon asset as an image.
 * @param name - File name inside the application `assets` directory.
 * @returns An image instance owned by the caller.
 */
export async function loadDesktopIcon(name: string): Promise<NativeImage> {
  return nativeImage.createFromBuffer(await readFile(desktopAssetPath(name)))
}

/**
 * Read the notification-area icon with its high-density representation.
 * @returns The tray image for the current display scale.
 */
export async function loadTrayIcon(): Promise<NativeImage> {
  const image = await loadDesktopIcon('tray.png')
  image.addRepresentation({ scaleFactor: 2, buffer: await readFile(desktopAssetPath('tray@2x.png')) })
  return image
}
