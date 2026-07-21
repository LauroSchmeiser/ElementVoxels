#include "SoundManager.h"
#include <iostream>
#include <algorithm>

SoundManager::SoundManager() {
}

SoundManager::~SoundManager() {
    shutdown();
}

bool SoundManager::init() {
    if (initialized) {
        return true;
    }

    int result = soloud.init(SoLoud::Soloud::CLIP_ROUNDOFF | SoLoud::Soloud::ENABLE_VISUALIZATION);
    if (result != SoLoud::SO_NO_ERROR) {
        std::cerr << "Failed to initialize SoLoud: " << result << std::endl;
        return false;
    }

    soloud.set3dSoundSpeed(343.0f);

    soloud.set3dListenerParameters(
            0.0f, 0.0f, 0.0f,   // position
            0.0f, 0.0f, 0.0f,   // velocity
            0.0f, 0.0f, -1.0f,  // at
            0.0f, 1.0f, 0.0f    // up
    );

    initialized = true;
    std::cout << "SoundManager initialized successfully." << std::endl;
    return true;
}

void SoundManager::shutdown() {
    if (!initialized) {
        return;
    }

    stopAllSounds();
    sounds.clear();
    activeHandles.clear();

    soloud.deinit();
    initialized = false;
    std::cout << "SoundManager shut down." << std::endl;
}

void SoundManager::update() {
    if (!initialized) {
        return;
    }

    soloud.update3dAudio();

    std::vector<SoLoud::handle> toRemove;
    for (const auto& [handle, id] : activeHandles) {
        if (!soloud.isValidVoiceHandle(handle)) {
            toRemove.push_back(handle);
        }
    }

    for (SoLoud::handle handle : toRemove) {
        activeHandles.erase(handle);
        if (handle == currentMusicHandle) {
            currentMusicHandle = 0;
        }
    }
}

bool SoundManager::loadSound(SoundID id, const std::string& filepath, SoundType type) {
    if (!initialized) {
        std::cerr << "SoundManager not initialized!" << std::endl;
        return false;
    }

    auto it = sounds.find(id);
    if (it != sounds.end() && it->second.isLoaded) {
        std::cout << "Sound already loaded: " << filepath << std::endl;
        return true;
    }

    try {
        SoundEntry entry;
        entry.type = type;
        entry.isLoaded = true;
        entry.filepath = filepath;

        if (type == SoundType::Music) {
            auto wavStream = std::make_unique<SoLoud::WavStream>();
            if (wavStream->load(filepath.c_str()) != SoLoud::SO_NO_ERROR) {
                std::cerr << "Failed to load music: " << filepath << std::endl;
                return false;
            }

            entry.source = std::move(wavStream);
        } else {
            auto wav = std::make_unique<SoLoud::Wav>();
            if (wav->load(filepath.c_str()) != SoLoud::SO_NO_ERROR) {
                std::cerr << "Failed to load sound: " << filepath << std::endl;
                return false;
            }
            entry.source = std::move(wav);
        }

        sounds[id] = std::move(entry);
        std::cout << "Loaded sound: " << filepath << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Exception loading sound: " << e.what() << std::endl;
        return false;
    }
}

bool SoundManager::loadMusic(SoundID id, const std::string& filepath) {
    return loadSound(id, filepath, SoundType::Music);
}

SoLoud::handle SoundManager::playSound(SoundID id, float volume, float pitch) {
    auto it = sounds.find(id);
    if (it == sounds.end() || !it->second.isLoaded) {
        std::cerr << "Sound not loaded: " << static_cast<int>(id) << std::endl;
        return 0;
    }

    return playSoundInternal(id, it->second.source.get(), it->second.type, volume, pitch, nullptr);
}

SoLoud::handle SoundManager::playSound3D(SoundID id, const glm::vec3& position,
                                         float volume, float pitch) {
    auto it = sounds.find(id);
    if (it == sounds.end() || !it->second.isLoaded) {
        std::cerr << "Sound not loaded: " << static_cast<int>(id) << std::endl;
        return 0;
    }

    return playSoundInternal(id, it->second.source.get(), it->second.type, volume, pitch, &position);
}

SoLoud::handle SoundManager::playMusic(SoundID id, bool loop, float volume) {
    auto it = sounds.find(id);
    if (it == sounds.end() || !it->second.isLoaded) {
        std::cerr << "Music not loaded: " << static_cast<int>(id) << std::endl;
        return 0;
    }

    if (currentMusicHandle != 0) {
        soloud.stop(currentMusicHandle);
        activeHandles.erase(currentMusicHandle);
        currentMusicHandle = 0;
    }

    it->second.source->setLooping(loop);

    SoLoud::handle handle = playSoundInternal(id, it->second.source.get(), SoundType::Music, volume, 1.0f, nullptr);

    currentMusicHandle = handle;
    currentMusicID = id;
    return handle;
}

SoLoud::handle SoundManager::playSoundInternal(SoundID id,
                                               SoLoud::AudioSource* source,
                                               SoundType type,
                                               float volume,
                                               float pitch,
                                               const glm::vec3* position) {
    if (!initialized || source == nullptr) {
        return 0;
    }

    float finalVolume = calculateFinalVolume(type, volume);
    SoLoud::handle handle = 0;

    if (position) {
        handle = soloud.play3d(*source,
                               position->x, position->y, position->z,
                               0.0f, 0.0f, 0.0f,
                               finalVolume,
                               true);
    } else {
        handle = soloud.play(*source, finalVolume, 0.0f, true);
    }

    if (handle == 0) {
        std::cerr << "Failed to play sound!" << std::endl;
        return 0;
    }

    if (pitch != 1.0f) {
        soloud.setRelativePlaySpeed(handle, pitch);
    }

    soloud.setPause(handle, false);

    activeHandles[handle] = id;
    return handle;
}

float SoundManager::calculateFinalVolume(SoundType type, float baseVolume) const {
    float finalVolume = baseVolume * masterVolume;

    switch (type) {
        case SoundType::Music:
            finalVolume *= musicVolume;
            break;
        case SoundType::SFX:
        case SoundType::UI:
            finalVolume *= sfxVolume;
            break;
    }

    return glm::clamp(finalVolume, 0.0f, 1.0f);
}

void SoundManager::stopSound(SoLoud::handle handle) {
    if (handle != 0) {
        soloud.stop(handle);
        activeHandles.erase(handle);
        if (handle == currentMusicHandle) {
            currentMusicHandle = 0;
        }
    }
}

void SoundManager::stopAllSounds() {
    soloud.stopAll();
    activeHandles.clear();
    currentMusicHandle = 0;
}

void SoundManager::stopMusic() {
    if (currentMusicHandle != 0) {
        SoLoud::handle oldHandle = currentMusicHandle;
        soloud.stop(oldHandle);
        activeHandles.erase(oldHandle);
        currentMusicHandle = 0;
    }
}

void SoundManager::stopSoundByID(SoundID id) {
    std::vector<SoLoud::handle> toRemove;
    for (const auto& [handle, soundID] : activeHandles) {
        if (soundID == id) {
            toRemove.push_back(handle);
        }
    }

    for (SoLoud::handle handle : toRemove) {
        soloud.stop(handle);
        activeHandles.erase(handle);
        if (handle == currentMusicHandle) {
            currentMusicHandle = 0;
        }
    }
}

void SoundManager::pauseSound(SoLoud::handle handle) {
    if (handle != 0) {
        soloud.setPause(handle, true);
    }
}

void SoundManager::resumeSound(SoLoud::handle handle) {
    if (handle != 0) {
        soloud.setPause(handle, false);
    }
}

void SoundManager::pauseAll() {
    soloud.setPauseAll(true);
}

void SoundManager::resumeAll() {
    soloud.setPauseAll(false);
}

void SoundManager::set3dListenerPosition(float x, float y, float z) {
    listenerPosition = glm::vec3(x, y, z);

    soloud.set3dListenerParameters(
            listenerPosition.x, listenerPosition.y, listenerPosition.z,
            listenerVelocity.x, listenerVelocity.y, listenerVelocity.z,
            listenerForward.x, listenerForward.y, listenerForward.z,
            listenerUp.x, listenerUp.y, listenerUp.z
    );
}

void SoundManager::set3dListenerPosition(const glm::vec3& position) {
    set3dListenerPosition(position.x, position.y, position.z);
}

void SoundManager::set3dListenerVelocity(const glm::vec3& velocity) {
    listenerVelocity = velocity;

    soloud.set3dListenerParameters(
            listenerPosition.x, listenerPosition.y, listenerPosition.z,
            listenerVelocity.x, listenerVelocity.y, listenerVelocity.z,
            listenerForward.x, listenerForward.y, listenerForward.z,
            listenerUp.x, listenerUp.y, listenerUp.z
    );
}

void SoundManager::set3dListenerParameters(const glm::vec3& position, const glm::vec3& velocity,
                                           const glm::vec3& forward, const glm::vec3& up) {
    listenerPosition = position;
    listenerVelocity = velocity;
    listenerForward = forward;
    listenerUp = up;

    soloud.set3dListenerParameters(
            position.x, position.y, position.z,
            velocity.x, velocity.y, velocity.z,
            forward.x, forward.y, forward.z,
            up.x, up.y, up.z
    );
}

void SoundManager::update3dAudio() {
    soloud.update3dAudio();
}

bool SoundManager::isSoundPlaying(SoLoud::handle handle) {
    if (handle == 0) return false;
    return soloud.isValidVoiceHandle(handle);
}

bool SoundManager::isSoundLoaded(SoundID id) const {
    auto it = sounds.find(id);
    return it != sounds.end() && it->second.isLoaded;
}

float SoundManager::getSoundVolume(SoLoud::handle handle) {
    if (handle == 0) return 0.0f;
    return soloud.getVolume(handle);
}

void SoundManager::setSoundVolume(SoLoud::handle handle, float volume) {
    if (handle != 0) {
        soloud.setVolume(handle, glm::clamp(volume, 0.0f, 1.0f));
    }
}

void SoundManager::setSoundPitch(SoLoud::handle handle, float pitch) {
    if (handle != 0) {
        soloud.setRelativePlaySpeed(handle, pitch);
    }
}

void SoundManager::setSound3DParameters(SoLoud::handle handle, const glm::vec3& position,
                                        float minDistance, float maxDistance) {
    if (handle != 0) {
        soloud.set3dSourcePosition(handle, position.x, position.y, position.z);
        soloud.set3dSourceMinMaxDistance(handle, minDistance, maxDistance);
    }
}

void SoundManager::setSound3DVelocity(SoLoud::handle handle, const glm::vec3& velocity) {
    if (handle != 0) {
        soloud.set3dSourceVelocity(handle, velocity.x, velocity.y, velocity.z);
    }
}

void SoundManager::setGlobalSoundVolume(float volume) {
    setMasterVolume(volume);
    refreshPlayingVolumes();
}

void SoundManager::setMasterVolume(float volume) {
    masterVolume = glm::clamp(volume, 0.0f, 1.0f);
    refreshPlayingVolumes();
}

void SoundManager::setSFXVolume(float volume) {
    sfxVolume = glm::clamp(volume, 0.0f, 1.0f);
    refreshPlayingVolumes();
}

void SoundManager::setMusicVolume(float volume) {
    musicVolume = glm::clamp(volume, 0.0f, 1.0f);
    refreshPlayingVolumes();
}

void SoundManager::refreshPlayingVolumes() {
    for (const auto& [handle, id] : activeHandles) {
        if (!soloud.isValidVoiceHandle(handle)) {
            continue;
        }

        auto it = sounds.find(id);
        if (it == sounds.end() || !it->second.isLoaded) {
            continue;
        }

        float volume = calculateFinalVolume(it->second.type, 1.0f);
        soloud.setVolume(handle, volume);
    }
}