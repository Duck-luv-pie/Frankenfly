# Imaging a card for the Pi from the Mac

Raspberry Pi Imager's command-line mode stalls on macOS (it cannot ask for admin rights), so:

1. Download the image: `curl -L -o os.img.xz https://downloads.raspberrypi.com/raspios_lite_arm64_latest`
2. Find the card: `diskutil list` (e.g. `/dev/disk8`, the built-in SDXC reader). It will be erased.
3. Write it (Terminal, asks for the Mac password): `xz -dc os.img.xz | sudo dd of=/dev/rdiskN bs=4m status=progress`
4. The Mac mounts the card's boot partition at `/Volumes/bootfs`. Replace its `user-data` and `network-config`
   (cloud-init; the shipped ones are commented templates) and `touch /Volumes/bootfs/ssh`:

```yaml
# user-data
#cloud-config
hostname: companion-pi
manage_etc_hosts: true
timezone: America/Toronto
ssh_pwauth: true
users:
- name: companion
  groups: users,adm,dialout,audio,netdev,video,plugdev,cdrom,games,input,gpio,spi,i2c,render,sudo
  shell: /bin/bash
  lock_passwd: false
  passwd: <openssl passwd -6 'the password'>
  sudo: ALL=(ALL) NOPASSWD:ALL
```

```yaml
# network-config (netplan v2; list every Wi-Fi the Pi may meet; iPhone hotspots use a curly apostrophe)
network:
  version: 2
  wifis:
    wlan0:
      dhcp4: true
      optional: true
      regulatory-domain: "CA"
      access-points:
        "Home network":
          password: "..."
```

5. `diskutil eject /dev/diskN`, card into the Pi (underside slot, label away from the board), power, ~3 min.
